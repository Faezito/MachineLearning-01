using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class CreditoPredictionResultModel
    {
        [ColumnName("PredictedLabel")]
        public bool PredicaoAprovado { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }
    }
}
