package org.rsultan.utils;

import java.io.File;

public class TestUtils {

  public static String getResourceFileName(String resourcePath) {
    var classLoader = CSVUtilsTest.class.getClassLoader();
    return new File(classLoader.getResource(resourcePath).getFile()).toString();
  }
}
