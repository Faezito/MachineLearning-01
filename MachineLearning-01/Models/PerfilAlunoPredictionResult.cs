using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class PerfilAlunoPredictionResult
    {
        [ColumnName("PredictedLabel")]
        public string PerfilPrevisto {  get; set; }
        public float[] Score {  get; set; }
    }
}
