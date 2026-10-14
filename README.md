# local-nural-net

Spring Boot 4.1.1 starter (Java 21, Maven).

## Requirements

- Java 21+
- Maven 3.6.3+ (or use the included wrapper)

## Run training UI

```bash
./mvnw spring-boot:run
```

Open [http://localhost:8080](http://localhost:8080) — hit **Train more** repeatedly to keep training the same nurons, **Stop** mid-run, **Generate** samples, or change the corpus.

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/status` | epochs, loss, running flag |
| `POST` | `/api/train` | `{ "epochs": 20, "learningRate": 0.3 }` — continues on current weights |
| `POST` | `/api/train/stop` | stop current run |
| `POST` | `/api/reset` | re-init weights |
| `POST` | `/api/generate` | `{ "seed": "hell", "length": 60 }` |
| `POST` | `/api/corpus` | `{ "text": "..." }` — replaces corpus (resets weights) |

## CLI next-char demo

```bash
./mvnw -q exec:java -Dexec.mainClass=com.localnuralnet.NextCharDemo
```

## Test

```bash
./mvnw test
```

## Package

```bash
./mvnw package
java -jar target/local-nural-net-0.0.1-SNAPSHOT.jar
```
