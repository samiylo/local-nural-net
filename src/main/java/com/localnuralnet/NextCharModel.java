package com.localnuralnet;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.BooleanSupplier;

/**
 * Minimal next-char model: one {@link Nuron} per vocabulary character (one-vs-rest).
 * Input is a flattened one-hot of the last {@code contextSize} characters.
 */
public class NextCharModel {

	private final int contextSize;
	private final char[] vocab;
	private final Map<Character, Integer> indexOf;
	private final Nuron[] outputs;

	public NextCharModel(String corpus, int contextSize) {
		this(corpus, contextSize, new Random(42));
	}

	public NextCharModel(String corpus, int contextSize, Random random) {
		if (contextSize < 1) {
			throw new IllegalArgumentException("contextSize must be at least 1");
		}
		this.contextSize = contextSize;
		this.indexOf = new LinkedHashMap<>();
		for (int i = 0; i < corpus.length(); i++) {
			indexOf.putIfAbsent(corpus.charAt(i), indexOf.size());
		}
		if (indexOf.isEmpty()) {
			throw new IllegalArgumentException("corpus must contain at least one character");
		}
		this.vocab = new char[indexOf.size()];
		for (Map.Entry<Character, Integer> e : indexOf.entrySet()) {
			vocab[e.getValue()] = e.getKey();
		}
		int inputSize = contextSize * vocab.length;
		this.outputs = new Nuron[vocab.length];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = new Nuron(inputSize, random);
		}
	}

	public void train(String corpus, int epochs, double learningRate) {
		train(corpus, epochs, learningRate, () -> false, null);
	}

	/**
	 * Runs more epochs on the same weights. Stops early when {@code shouldStop} is true.
	 * {@code onEpoch} receives (1-based epoch in this run, avg loss).
	 */
	public void train(
			String corpus,
			int epochs,
			double learningRate,
			BooleanSupplier shouldStop,
			EpochListener onEpoch) {
		if (corpus.length() <= contextSize) {
			throw new IllegalArgumentException("corpus must be longer than contextSize");
		}
		for (int epoch = 0; epoch < epochs; epoch++) {
			if (shouldStop != null && shouldStop.getAsBoolean()) {
				return;
			}
			double avg;
			synchronized (this) {
				double loss = 0;
				int steps = 0;
				for (int i = 0; i + contextSize < corpus.length(); i++) {
					if (shouldStop != null && shouldStop.getAsBoolean()) {
						return;
					}
					double[] input = encodeContext(corpus, i);
					int target = index(corpus.charAt(i + contextSize));
					for (int c = 0; c < outputs.length; c++) {
						double y = outputs[c].activate(input);
						double t = (c == target) ? 1.0 : 0.0;
						loss += (t - y) * (t - y);
						outputs[c].train(t, learningRate);
					}
					steps++;
				}
				avg = loss / steps;
			}
			if (onEpoch != null) {
				onEpoch.onEpoch(epoch + 1, avg);
			}
		}
	}

	/** Greedy generation: always pick the highest-scoring next char. */
	public synchronized String generate(String seed, int length) {
		StringBuilder out = new StringBuilder(seed);
		while (out.length() < contextSize) {
			out.insert(0, vocab[0]);
		}
		for (int n = 0; n < length; n++) {
			int start = out.length() - contextSize;
			double[] input = encodeContext(out, start);
			int best = 0;
			double bestScore = Double.NEGATIVE_INFINITY;
			for (int c = 0; c < outputs.length; c++) {
				double score = outputs[c].activate(input);
				if (score > bestScore) {
					bestScore = score;
					best = c;
				}
			}
			out.append(vocab[best]);
		}
		return out.substring(Math.min(seed.length(), out.length()));
	}

	public int vocabSize() {
		return vocab.length;
	}

	public int contextSize() {
		return contextSize;
	}

	private double[] encodeContext(CharSequence text, int start) {
		double[] input = new double[contextSize * vocab.length];
		for (int t = 0; t < contextSize; t++) {
			int idx = index(text.charAt(start + t));
			input[t * vocab.length + idx] = 1.0;
		}
		return input;
	}

	private int index(char ch) {
		Integer idx = indexOf.get(ch);
		if (idx == null) {
			return 0;
		}
		return idx;
	}

	@FunctionalInterface
	public interface EpochListener {
		void onEpoch(int epochInRun, double avgLoss);
	}
}
