using MachineLearning_01.Models;
using Microsoft.ML;

namespace MachineLearning_01.ML
{
    public class CasaModelPredictor
    {
        private MLContext _mlContext = new();
        private ITransformer _modeloCarregado;

        public void CarregarModelo(string caminho)
        {
            DataViewSchema modeloSchema;
            _modeloCarregado = _mlContext.Model.Load(caminho, out modeloSchema);
        }

        public CasaPredictionResult Prever(CasaInputData novaCasa)
        {
            var predictionEngine = _mlContext.Model
                .CreatePredictionEngine<CasaInputData, CasaPredictionResult>(_modeloCarregado);

            return predictionEngine.Predict(novaCasa);
        }
    }
}
