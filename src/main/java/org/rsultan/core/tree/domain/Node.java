package org.rsultan.core.tree.domain;

public record Node(
    int feature,
    double featureThreshold,
    int predictedLabel,
    Node left,
    Node right
) {}
