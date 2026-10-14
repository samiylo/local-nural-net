package com.localnuralnet;

import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("/api")
public class TrainingController {

	private final TrainingService training;

	public TrainingController(TrainingService training) {
		this.training = training;
	}

	@GetMapping("/status")
	public TrainingService.Status status() {
		return training.status();
	}

	@PostMapping("/train")
	public TrainingService.Status train(@RequestBody(required = false) TrainRequest body) {
		int epochs = body != null && body.epochs() != null ? body.epochs() : 20;
		double lr = body != null && body.learningRate() != null ? body.learningRate() : 0.3;
		try {
			return training.startTraining(epochs, lr);
		} catch (IllegalArgumentException e) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());
		}
	}

	@PostMapping("/train/stop")
	public TrainingService.Status stop() {
		return training.stopTraining();
	}

	@PostMapping("/reset")
	public TrainingService.Status reset() {
		try {
			return training.reset();
		} catch (IllegalStateException e) {
			throw new ResponseStatusException(HttpStatus.CONFLICT, e.getMessage());
		}
	}

	@PostMapping("/corpus")
	public TrainingService.Status corpus(@RequestBody CorpusRequest body) {
		try {
			return training.setCorpus(body.text());
		} catch (IllegalStateException e) {
			throw new ResponseStatusException(HttpStatus.CONFLICT, e.getMessage());
		} catch (IllegalArgumentException e) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());
		}
	}

	@PostMapping("/generate")
	public Map<String, String> generate(@RequestBody(required = false) GenerateRequest body) {
		String seed = body != null ? body.seed() : null;
		int length = body != null && body.length() != null ? body.length() : 60;
		return Map.of("text", training.generate(seed, length));
	}

	public record TrainRequest(Integer epochs, Double learningRate) {
	}

	public record CorpusRequest(String text) {
	}

	public record GenerateRequest(String seed, Integer length) {
	}
}
