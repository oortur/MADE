package linreg

import breeze.linalg._
import breeze.numerics.pow

object Main {
  def fit(x: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    inv(x.t * x) * x.t * y
  }

  def mse(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    sum(pow(yTrue - yPred, 2))
  }

  def crossVal(x: DenseMatrix[Double], y: DenseVector[Double], nFolds: Int): DenseVector[Double] = {
    val size = x.rows

    def makeIdsCrossVal(size: Int, nFolds: Int): Seq[(Seq[Int], Seq[Int])] = {
      val shuffledIndices: Seq[Int] = scala.util.Random.shuffle(Range(0, size))
      val foldSize = size / nFolds
      var ans: Seq[(Seq[Int], Seq[Int])] = Seq()
      for (i <- 0 until nFolds) {
        val testIndices = shuffledIndices.slice(foldSize * i, foldSize * (i + 1))
        val trainIndicesLeft = shuffledIndices.slice(0, foldSize * i)
        val trainIndicesRight = shuffledIndices.slice(foldSize * (i + 1), size)
        val trainIndices = trainIndicesLeft ++ trainIndicesRight
        ans ++= Seq(Tuple2(trainIndices, testIndices))
      }
      return ans
    }

    val idsCrossVal = makeIdsCrossVal(size, nFolds)
    var bestError = sum(y)
    var bestW = DenseVector.zeros[Double](size)
    for (i <- 0 until nFolds) {
      var pair = idsCrossVal(i)
      var trainIndices: IndexedSeq[Int] = pair._1.toIndexedSeq
      var testIndices: IndexedSeq[Int] = pair._2.toIndexedSeq
      var xFoldTrain: DenseMatrix[Double] = x(trainIndices, ::).toDenseMatrix
      var xFoldTest: DenseMatrix[Double]= x(testIndices, ::).toDenseMatrix
      var yFoldTrain: DenseVector[Double] = y(trainIndices).toDenseVector
      var yFoldTest: DenseVector[Double] = y(testIndices).toDenseVector
      var w = fit(xFoldTrain, yFoldTrain)
      var yFoldPred = xFoldTest * w
      var error = mse(yFoldTest, yFoldPred)
      println(s"Fold ${i}: MSE: ${error}")
      bestW = if (error < bestError) w else bestW
      bestError = if (error < bestError) error else bestError
    }
    println(s"Best MSE: ${bestError}")
    return bestW
  }

  def predict(x: DenseMatrix[Double], w: DenseVector[Double]): DenseVector[Double] = {
    x * w
  }

  def main(args: Array[String]): Unit = {
    val trainFileName = "data/train.csv"
    val testFileName = "data/test.csv"
    val predFileName = "data/prediction.csv"
    if (args.length == 3) {
      val trainFileName = args(0)
      val testFileName = args(1)
      val predFileName = args(2)
    }

    val trainFile = new java.io.File(trainFileName)
    val train = csvread(trainFile)
    val xTrain = train(::, 0 to -2)
    val yTrain = train(::, -1)

    val testFile = new java.io.File(testFileName)
    val xTest = csvread(testFile)

    val w = crossVal(xTrain, yTrain, nFolds=5)
    val pred = predict(xTest, w).toDenseMatrix.t

    val predFile = new java.io.File(predFileName)
    csvwrite(predFile, pred)
  }
}
