using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class CasaPredictionResult // CLASSE DE SAÍDA DE DADOS, DA PREDIÇÃO DO MODELO DE ML
    {
        [ColumnName("Score")] // SCORE é onde o ML.NET colocará o resultado
        public float PrecoPrevisto { get; set; }
    }
}
