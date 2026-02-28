from __future__ import annotations

from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class TextStatsTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """Adds domain-specific text statistics from raw review text."""

    inputCol = Param(Params._dummy(), "inputCol", "Input text column")
    outputPrefix = Param(Params._dummy(), "outputPrefix", "Prefix for output columns")

    def __init__(self, inputCol: str = "review_text", outputPrefix: str = "txt_"):
        super().__init__()
        self._setDefault(inputCol="review_text", outputPrefix="txt_")
        self.setParams(inputCol=inputCol, outputPrefix=outputPrefix)

    def setParams(self, inputCol: str = "review_text", outputPrefix: str = "txt_"):
        return self._set(inputCol=inputCol, outputPrefix=outputPrefix)

    def _transform(self, df: DataFrame) -> DataFrame:
        c = self.getOrDefault(self.inputCol)
        p = self.getOrDefault(self.outputPrefix)
        text = F.coalesce(F.col(c), F.lit(""))

        return (
            df.withColumn(f"{p}token_count", F.size(F.split(text, r"\\s+")))
            .withColumn(f"{p}char_count", F.length(text))
            .withColumn(f"{p}exclaim_count", F.length(text) - F.length(F.regexp_replace(text, "!", "")))
            .withColumn(
                f"{p}upper_ratio",
                F.when(F.length(text) == 0, F.lit(0.0)).otherwise(
                    F.length(F.regexp_replace(text, r"[^A-Z]", "")) / F.length(text)
                ),
            )
        )
