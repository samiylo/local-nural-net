package com.localnuralnet;

import java.util.Arrays;
import java.util.Random;

/**
 * A single nuron: weighted sum of inputs plus bias, then an activation.
 */
public class Nuron {

	private final double[] weights;
	private double bias;
	private double lastOutput;
	private double[] lastInputs;

	public Nuron(int inputCount) {
		this(inputCount, new Random());
	}

	public Nuron(int inputCount, Random random) {
		if (inputCount < 1) {
			throw new IllegalArgumentException("inputCount must be at least 1");
		}
		this.weights = new double[inputCount];
		for (int i = 0; i < inputCount; i++) {
			// Small random weights in [-0.5, 0.5)
			weights[i] = random.nextDouble() - 0.5;
		}
		this.bias = random.nextDouble() - 0.5;
	}

	public Nuron(double[] weights, double bias) {
		this.weights = Arrays.copyOf(weights, weights.length);
		this.bias = bias;
	}

	/** Forward pass: sigmoid(dot(inputs, weights) + bias). */
	public double activate(double[] inputs) {
		if (inputs.length != weights.length) {
			throw new IllegalArgumentException(
					"Expected " + weights.length + " inputs, got " + inputs.length);
		}
		lastInputs = Arrays.copyOf(inputs, inputs.length);
		double sum = bias;
		for (int i = 0; i < weights.length; i++) {
			sum += weights[i] * inputs[i];
		}
		lastOutput = sigmoid(sum);
		return lastOutput;
	}

	/**
	 * One step of gradient descent for a supervised target.
	 * Call {@link #activate(double[])} first so lastInputs/lastOutput are set.
	 */
	public void train(double target, double learningRate) {
		if (lastInputs == null) {
			throw new IllegalStateException("Call activate() before train()");
		}
		double error = target - lastOutput;
		double delta = error * sigmoidDerivative(lastOutput);
		for (int i = 0; i < weights.length; i++) {
			weights[i] += learningRate * delta * lastInputs[i];
		}
		bias += learningRate * delta;
	}

	public double[] getWeights() {
		return Arrays.copyOf(weights, weights.length);
	}

	public double getBias() {
		return bias;
	}

	public double getLastOutput() {
		return lastOutput;
	}

	private static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	private static double sigmoidDerivative(double sigmoidOutput) {
		return sigmoidOutput * (1.0 - sigmoidOutput);
	}

	@Override
	public String toString() {
		return "Nuron{weights=" + Arrays.toString(weights) + ", bias=" + bias + "}";
	}
}
