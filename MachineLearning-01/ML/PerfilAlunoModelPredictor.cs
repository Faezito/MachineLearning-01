using MachineLearning_01.Models;
using Microsoft.ML;

namespace MachineLearning_01.ML
{
    public class PerfilAlunoModelPredictor
    {
        private MLContext context = new();
        private ITransformer modeloCarregado;

        public void CarregarModelo(string caminho)
        {
            DataViewSchema modeloSchema;
            modeloCarregado = context.Model.Load(caminho, out modeloSchema);
        }

        public PerfilAlunoPredictionResult Prever(PerfilAlunoInputDataModel novoAluno)
        {
            var predictionEngine = context.Model
                .CreatePredictionEngine<PerfilAlunoInputDataModel, PerfilAlunoPredictionResult>
                (modeloCarregado);

            return predictionEngine.Predict(novoAluno);
        }
    }
}
