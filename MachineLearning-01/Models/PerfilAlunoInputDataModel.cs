using Microsoft.ML.Data;

namespace MachineLearning_01.Models
{
    public class PerfilAlunoInputDataModel
    {
        [LoadColumn(0)]
        public float NotaProficienciaGramatical {  get; set; }
        [LoadColumn(1)]
        public float NotaCompreensaoOral {  get; set; }
        [LoadColumn(2)]
        public float NotaConversacao {  get; set; }
        [LoadColumn(3)]
        public string PerfilAluno {  get; set; }
    }
}
