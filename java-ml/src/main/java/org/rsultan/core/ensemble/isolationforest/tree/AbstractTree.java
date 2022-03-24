package org.rsultan.core.ensemble.isolationforest.tree;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ensemble.isolationforest.domain.IsolationNode;

import static org.rsultan.core.ensemble.isolationforest.utils.ScoreUtils.averagePathLength;

public abstract class AbstractTree<NODE_DATA> {

    protected final int treeDepthLimit;
    protected IsolationNode<NODE_DATA> tree;

    public AbstractTree(int treeDepthLimit) {
        this.treeDepthLimit = treeDepthLimit;
    }

    protected IsolationNode<NODE_DATA> buildTree(INDArray X, int currentDepth){
        if (currentDepth <= 0 || X.rows() <= 2) {
            return new IsolationNode<>(X);
        }
        return buildNode(X, currentDepth);
    }

    protected abstract IsolationNode<NODE_DATA> buildNode(INDArray X, int currentDepth);

    protected abstract boolean chooseLeftNode(INDArray row, NODE_DATA slope);

    public INDArray predict(INDArray matrix) {
        var pathLengths = Nd4j.zeros(1, matrix.rows());
        for (int i = 0; i < matrix.rows(); i++) {
            var row = matrix.getRow(i);
            var node = tree;
            int length = 0;
            while (!node.isLeaf()) {
                var slope = node.nodeData();
                node = chooseLeftNode(row, slope) ? node.left() : node.right();
                length++;
            }
            int leafSize = node.data().rows();
            pathLengths.put(0, i, length + averagePathLength(leafSize));
        }
        return pathLengths;
    }
}
