using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class CasaInputData // CLASSE DE ENTRADA DE DADOS
    {
        [LoadColumn(0)]
        public float Tamanho { get; set; }
        [LoadColumn(1)]
        public float Quartos { get; set; }
        [LoadColumn(2)]
        public float Preco { get; set; }
    }
}
