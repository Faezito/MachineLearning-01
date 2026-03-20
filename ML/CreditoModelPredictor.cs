using MachineLearning_01.Models;
using Microsoft.ML;

namespace MachineLearning_01.ML
{
    public class CreditoModelPredictor
    {
        private MLContext mLContext = new();
        private ITransformer modeloCarregado;

        public void CarregarModelo(string caminho)
        {
            DataViewSchema modeloSchema;
            modeloCarregado = mLContext.Model.Load(caminho, out modeloSchema);
        }

        public CreditoPredictionResultModel Prever(CreditoInputDataModel creditoNovo)
        {
            var predictionEngine = mLContext.Model.CreatePredictionEngine<CreditoInputDataModel, CreditoPredictionResultModel>(modeloCarregado);

            return predictionEngine.Predict(creditoNovo);
        }

    }
}
