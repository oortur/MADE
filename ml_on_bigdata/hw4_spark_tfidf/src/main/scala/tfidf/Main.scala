package tfidf

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]): Unit = {
    // create spark session
    val spark = SparkSession.builder()
      // master address
      .master("local[*]")
      // app name in spark interface
      .appName("made-demo")
      // use current or create new
      .getOrCreate()

    import spark.implicits._

    // read dataset
    // https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("tripadvisor_hotel_reviews.csv")

    val corpusSize = df.count()
    val N_TOP_TERMS = 100

    val docTerms = df
      .select(
        lower(
          trim(
            regexp_replace($"Review", "[\\p{Punct}]", "")
          )
        ).alias("text")
      )
      .withColumn("doc_id", monotonically_increasing_id())
      .withColumn("term", explode(split($"text", " ")))

    val docSize = docTerms
      .groupBy("doc_id")
      .count()
      .withColumnRenamed("count", "doc_size")

    val wc = docTerms
      .groupBy("doc_id", "term")
      .count()

    val topTerms = wc
      .groupBy("term")
      .sum("count")
      .sort(desc("sum(count)"))
      .limit(N_TOP_TERMS)
    // list of top frequent terms in dataset
    val topTermsList = topTerms.select("term").collect().map(_(0)).toList

    val tf = wc
      .join(
        docSize.select("doc_id", "doc_size"), "doc_id"
      )
      .withColumn("tf", col("count") / col("doc_size"))

    val idf = wc
      .groupBy("term")
      .count()
      .withColumnRenamed("count", "doc_freq")
      .withColumn("idf", log(lit(corpusSize.toDouble) / col("doc_freq")))

    val tfidf = tf
      .select("doc_id", "term", "tf")
      .where(col("term").isInCollection(topTermsList))
      .join(idf.select("term", "idf"), "term")
      .withColumn("tfidf", col("tf") * col("idf"))
      .sort("doc_id", "term")
      .drop("tf", "idf")

    tfidf.show

    val tfidfTable = tfidf
      .groupBy("doc_id")
      .pivot("term")
      .sum("tfidf")
      .na.fill(0)

    tfidfTable.show

    // save tfidf table to csv file if needed
    // tfidfTable.write.csv("tfidf.csv")
  }
}
