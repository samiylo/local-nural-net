package com.localnuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.springframework.stereotype.Service;

@Service
public class TrainingService {

	private static final int CONTEXT_SIZE = 4;
	private static final int MAX_LOSS_HISTORY = 200;

	private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
		Thread t = new Thread(r, "nuron-trainer");
		t.setDaemon(true);
		return t;
	});

	private final AtomicBoolean running = new AtomicBoolean(false);
	private final AtomicBoolean stopRequested = new AtomicBoolean(false);
	private final AtomicInteger totalEpochs = new AtomicInteger(0);
	private final AtomicReference<Double> lastLoss = new AtomicReference<>(null);
	private final AtomicReference<String> lastError = new AtomicReference<>(null);
	private final List<Double> lossHistory = Collections.synchronizedList(new ArrayList<>());

	private volatile String corpus = Corpus.DEFAULT;
	private volatile NextCharModel model = new NextCharModel(Corpus.DEFAULT, CONTEXT_SIZE);
	private volatile double lastLearningRate = 0.3;

	public synchronized Status startTraining(int epochs, double learningRate) {
		if (epochs < 1) {
			throw new IllegalArgumentException("epochs must be at least 1");
		}
		if (!running.compareAndSet(false, true)) {
			return status();
		}
		stopRequested.set(false);
		lastError.set(null);
		lastLearningRate = learningRate;
		final String text = corpus;
		final NextCharModel current = model;

		executor.submit(() -> {
			try {
				current.train(text, epochs, learningRate, stopRequested::get, (epochInRun, avgLoss) -> {
					totalEpochs.incrementAndGet();
					lastLoss.set(avgLoss);
					lossHistory.add(avgLoss);
					while (lossHistory.size() > MAX_LOSS_HISTORY) {
						lossHistory.remove(0);
					}
				});
			} catch (Exception e) {
				lastError.set(e.getMessage());
			} finally {
				running.set(false);
				stopRequested.set(false);
			}
		});
		return status();
	}

	public Status stopTraining() {
		stopRequested.set(true);
		return status();
	}

	public synchronized Status reset() {
		if (running.get()) {
			throw new IllegalStateException("Stop training before reset");
		}
		model = new NextCharModel(corpus, CONTEXT_SIZE);
		totalEpochs.set(0);
		lastLoss.set(null);
		lastError.set(null);
		lossHistory.clear();
		return status();
	}

	public synchronized Status setCorpus(String newCorpus) {
		if (running.get()) {
			throw new IllegalStateException("Stop training before changing corpus");
		}
		if (newCorpus == null || newCorpus.isBlank()) {
			throw new IllegalArgumentException("corpus must not be blank");
		}
		corpus = newCorpus;
		model = new NextCharModel(corpus, CONTEXT_SIZE);
		totalEpochs.set(0);
		lastLoss.set(null);
		lastError.set(null);
		lossHistory.clear();
		return status();
	}

	public String generate(String seed, int length) {
		if (seed == null || seed.isBlank()) {
			seed = "hell";
		}
		if (length < 1) {
			length = 40;
		}
		return seed + model.generate(seed, length);
	}

	public Status status() {
		return new Status(
				running.get(),
				totalEpochs.get(),
				lastLoss.get(),
				lastLearningRate,
				model.vocabSize(),
				model.contextSize(),
				corpus.length(),
				List.copyOf(lossHistory),
				lastError.get());
	}

	public record Status(
			boolean running,
			int totalEpochs,
			Double lastLoss,
			double learningRate,
			int vocabSize,
			int contextSize,
			int corpusLength,
			List<Double> lossHistory,
			String error) {
	}
}
