package com.localnuralnet;

/**
 * CLI trainer (same model as the web UI).
 *
 * <pre>
 *   ./mvnw -q exec:java -Dexec.mainClass=com.localnuralnet.NextCharDemo
 * </pre>
 */
public class NextCharDemo {

	public static void main(String[] args) {
		int contextSize = 4;
		int epochs = 80;
		double learningRate = 0.3;

		NextCharModel model = new NextCharModel(Corpus.DEFAULT, contextSize);
		System.out.println("vocab size: " + model.vocabSize());
		model.train(Corpus.DEFAULT, epochs, learningRate);

		String seed = "hell";
		String generated = model.generate(seed, 60);
		System.out.println("seed:      " + seed);
		System.out.println("generated: " + seed + generated);
	}
}
