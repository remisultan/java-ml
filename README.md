# java-ml
This repo intends to implement Machine Learning algorithms with Java and https://github.com/deeplearning4j/nd4j purely for understanding

You can check for examples in this package: ``src/main/java/org/rsultan/example``

## Requirements

- JDK15+

## Getting started

Add this repository to your `pom.xml`
```xml
 <repositories>
    <repository>
        <id>release.archiva.rsultan.org</id>
        <name>Java ML repository</name>
        <url>https://archiva.rsultan.org/repository/internal</url>
    </repository>
 </repositories>
```

And then add the corresponding dependency 
```xml
    <dependency>
      <groupId>org.rsultan</groupId>
      <artifactId>java-ml</artifactId>
      <version>[latest-version]</version>
    </dependency>
```

You can also clone the repo and add the dependency according to your version

```bash
 $ git fetch --all --tags
 $ git checkout tags/<latest-release> -b <your-branch>  
 $ ./mvnw clean install
```

Once your are good with this, you can go read the [wiki](https://github.com/remisultan/java-ml/wiki)

You can also check the examples in the `java-ml-example` module.

Good luck !