# java-ml
This repo intends to implement Machine Learning algorithms with Java and https://github.com/deeplearning4j/nd4j purely for understanding

You can check for examples in this package: ``src/main/java/org/rsultan/example``

## Getting started

Clone the repo and execute this

```bash
 $ git fetch --all --tags
 $ git checkout tags/<latest-release> -b <your-branch>  
 $ ./mvnw clean install
```

Then you can import this to your `pom.xml`

```xml
    <dependency>
      <groupId>org.rsultan</groupId>
      <artifactId>java-ml</artifactId>
      <version>[latest-version]</version>
    </dependency>
```

There is an existing artifactory today but I am not satisfied with how to make it public today.
There will be an artifactory in the future.

Once your are good with this, you can go read the [wiki](https://github.com/remisultan/java-ml/wiki)

Good luck !