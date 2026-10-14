package com.localnuralnet;


public class NextCharDemo {

	private static final String CORPUS = """
			hello world. hello there. hello friend.
			the cat sat on the mat. the dog sat on the log.
			once upon a time there was a cat. once upon a time there was a dog.
			""";

	public static void main(String[] args) {
		int contextSize = 10;
		int epochs = 500;
		double learningRate = 0.3;

		NextCharModel model = new NextCharModel(CORPUS, contextSize);
		System.out.println("vocab size: " + model.vocabSize());
		model.train(CORPUS, epochs, learningRate);

		String seed = "mat";
		String generated = model.generate(seed, 60);
		System.out.println("seed:      " + seed);
		System.out.println("generated: " + seed + generated);
	}
}
