package org.rsultan.core;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.rsultan.core.serde.GzipSerDe;

public class Models {

  public static <T> T read(String pathName) throws IOException, ClassNotFoundException {
    var modelPath = Paths.get(pathName);
    return read(modelPath);
  }

  public static <T> T read(Path modelPath) throws IOException, ClassNotFoundException {
    byte[] fileBytes = Files.readAllBytes(modelPath);
    return new GzipSerDe<T>().deserialise(fileBytes);
  }

  public static <T> void write(String pathName, T object) throws IOException {
    var modelPath = Paths.get(pathName);
    write(modelPath, object);
  }

  public static <T> void write(Path path, T object) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(path.toAbsolutePath().toString())) {
      var objectToBytes = new GzipSerDe<T>().serialise(object);
      fos.write(objectToBytes);
    }
  }
}
