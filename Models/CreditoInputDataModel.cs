using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class CreditoInputDataModel // classe de entrada do csv
    {
        [LoadColumn(0)]
        public float RendaMensal {  get; set; }
        [LoadColumn(1)]
        public float EstadoCivil { get; set; } // 0 - solteiro, 1 - casado, 2 - divorciado, 3 - viuvo, 4 - UE
        [LoadColumn(2)]
        public float NumeroDependentes { get; set; }
        [LoadColumn(3)]
        public float PossuiVeiculo { get; set; }
        [LoadColumn(4)]
        public float JaNegadoAntes { get; set; }
        [LoadColumn(5)]
        public bool Aprovado {  get; set; }

    }
}
