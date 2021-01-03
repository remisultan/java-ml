package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.KMeans;
import org.rsultan.core.regression.impl.LinearRegression;
import org.rsultan.dataframe.Dataframes;

import java.io.IOException;

public class KmeansExample {

    /*
     The data used is the infamous IRIS dataset
     Make sure args[0] /path/to/your/src/main/resources/softmax/iris.data
    */
    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var df = Dataframes.csv(args[0], ",", false);
        var kMeans = new KMeans(3, 10).train(df.withoutColumn("c4"));

        System.out.println("Centers: " + kMeans.getC());
        System.out.println("Error: " + kMeans.getError());
    }
}
