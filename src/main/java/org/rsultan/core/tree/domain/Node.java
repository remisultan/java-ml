package org.rsultan.core.tree.domain;

public record Node(
    int feature,
    double featureThreshold,
    Number predictedResponse,
    Node left,
    Node right
) {}
