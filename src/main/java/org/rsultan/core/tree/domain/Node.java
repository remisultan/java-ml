package org.rsultan.core.tree.domain;

import java.io.Serializable;

public record Node(
    int feature,
    double featureThreshold,
    Number predictedResponse,
    Node left,
    Node right
) implements Serializable {}
