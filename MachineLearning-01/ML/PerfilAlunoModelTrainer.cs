using MachineLearning_01.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MachineLearning_01.ML
{
    public class PerfilAlunoModelTrainer
    {
        private MLContext mLContext = new();
        private IDataView dados;
        private ITransformer modeloTreinado;

        public void CarregarDadosCSV(string path)
        {
            dados = mLContext.Data.LoadFromTextFile<PerfilAlunoInputDataModel>(
                    path: path,
                    hasHeader: true,
                    separatorChar: ','
                );
        }

        public void TreinarModelo()
        {
            var pipeline = mLContext.Transforms.Conversion.MapValueToKey(
                    "Label",
                    nameof(PerfilAlunoInputDataModel.PerfilAluno)
                    )
                    .Append(mLContext.Transforms.Concatenate(
                    "Features",
                    nameof(PerfilAlunoInputDataModel.NotaCompreensaoOral),
                    nameof(PerfilAlunoInputDataModel.NotaConversacao),
                    nameof(PerfilAlunoInputDataModel.NotaProficienciaGramatical)
                ))
                .Append(mLContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
                ).Append(mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")
                );

            // treinar modelo
            modeloTreinado = pipeline.Fit(dados);
        }

        public void SalvarModelo(string path)
        {
            mLContext.Model.Save(modeloTreinado, dados.Schema, path);
        }

        public void AvaliarModelo()
        {
            var previsoes = modeloTreinado.Transform(dados);

            var metricas = mLContext.MulticlassClassification.Evaluate(data: previsoes, labelColumnName: "Label");
            Console.WriteLine($"MicroAccuracy: {metricas.MicroAccuracy:P2}"); //Quanto ele acertou no geral, mas pode errar caso hajam muitos de um e poucos de outro (maior = melhor)
            Console.WriteLine($"MacroAccuracy: {metricas.MacroAccuracy:P2}"); // Média de acerto para cada grupo de resultados (maior = melhor)
            Console.WriteLine($"Logloss: {metricas.LogLoss:P2}"); // Certeza do resultado, quanto menor, melhor
            Console.WriteLine($"LoglossReduction: {metricas.LogLossReduction:P2}"); // O quanto ele aprendeu, chutou menos, maior = melhor

            // Os dois primeiros são relativos à quantidade de cada grupo (intermediario, iniciante, avancado), ou seja, quanto mais dados de um grupo, maior a chance de acertar, quanto menor, menor a chance
        }

        public void AvaliarMelhorModelo()
        {
            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 60
            };

            var experiment = mLContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
            Console.WriteLine("Avaliando modelo...");
            var res = experiment.Execute(dados, labelColumnName: nameof(PerfilAlunoInputDataModel.PerfilAluno));

            var melhor = res.BestRun;

            Console.WriteLine($"Melhor algoritmo: {melhor.TrainerName}");
            Console.WriteLine($"MicroAccuracy: {melhor.ValidationMetrics.MicroAccuracy:P2}"); //Quanto ele acertou no geral, mas pode errar caso hajam muitos de um e poucos de outro (maior = melhor)
            Console.WriteLine($"MacroAccuracy: {melhor.ValidationMetrics.MacroAccuracy:P2}"); // Média de acerto para cada grupo de resultados (maior = melhor)
            Console.WriteLine($"Logloss: {melhor.ValidationMetrics.LogLoss:P2}"); // Certeza do resultado, quanto menor, melhor
            Console.WriteLine($"LoglossReduction: {melhor.ValidationMetrics.LogLossReduction:P2}"); // O quanto ele aprendeu, chutou menos, maior = melhor

            modeloTreinado = melhor.Model;
        }
    }
}
