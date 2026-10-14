# local-nural-net

Spring Boot 4.1.1 starter (Java 21, Maven).

## Requirements

- Java 21+
- Maven 3.6.3+ (or use the included wrapper)

## Run Spring Boot

```bash
./mvnw spring-boot:run
```

## Next-char trainer (core Java on `Nuron`)

Trains one `Nuron` per character to predict the next char from a short context window.

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
