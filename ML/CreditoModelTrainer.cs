using MachineLearning_01.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MachineLearning_01.ML
{
    public class CreditoModelTrainer
    {
        private MLContext _mlContext = new();
        private IDataView _dados;
        private ITransformer _modeloTreinado;

        public void CarregarDadosCSV(string caminho)
        {
            _dados = _mlContext.Data.LoadFromTextFile<CreditoInputDataModel>(
                    path: caminho,
                    hasHeader: true,
                    separatorChar: ','
                );
        }

        public void TreinarModelo()
        {
            var pipeline = _mlContext.Transforms.Concatenate(
                "Features",
                nameof(CreditoInputDataModel.RendaMensal),
                nameof(CreditoInputDataModel.PossuiVeiculo),
                nameof(CreditoInputDataModel.EstadoCivil),
                nameof(CreditoInputDataModel.NumeroDependentes),
                nameof(CreditoInputDataModel.JaNegadoAntes)
                )
                //.Append(_mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                //        labelColumnName: "Aprovado"
                //    )
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                        labelColumnName: "Aprovado"
                    )
                );

            _modeloTreinado = pipeline.Fit(_dados);
        }

        public void SalvarModelo(string caminho)
        {
            _mlContext.Model.Save(_modeloTreinado, _dados.Schema, caminho);
        }

        public void AvaliarModelo()
        {
            var previsoes = _modeloTreinado.Transform(_dados);

            var metricas = _mlContext.BinaryClassification.Evaluate(
                    data: previsoes,
                    labelColumnName: nameof(CreditoInputDataModel.Aprovado)
                );

            AvaliacaoModel avaliacao = new()
            {
                Acuracia = metricas.Accuracy,
                Precisao = metricas.PositivePrecision,
                Recall = metricas.PositiveRecall,
                F1Score = metricas.F1Score

            };

            Console.WriteLine($"Acuracia: {avaliacao.Acuracia:P2} \nPrecisão: {avaliacao.Precisao:P2} \nRecall: {avaliacao.Recall:P2}\nF1Score: {avaliacao.F1Score:P2}\n");
        }

        public void EncontrarMelhorAlgoritmo()
        {
            BinaryExperimentSettings experimentSettings = new()
            {
                MaxExperimentTimeInSeconds = 60
            };

            var experiment = _mlContext.Auto()
                .CreateBinaryClassificationExperiment(experimentSettings);

            Console.WriteLine("Encontrando melhor algoritmo...");

            var res = experiment.Execute(
                _dados, 
                labelColumnName: nameof(CreditoInputDataModel.Aprovado)
                );

            var melhorResultado = res.BestRun;

            AvaliacaoModel avaliacao = new()
            {
                Acuracia = melhorResultado.ValidationMetrics.Accuracy,
                Precisao = melhorResultado.ValidationMetrics.PositivePrecision,
                Recall = melhorResultado.ValidationMetrics.PositiveRecall,
                F1Score = melhorResultado.ValidationMetrics.F1Score

            };

            Console.WriteLine($"Acuracia: {avaliacao.Acuracia:P2} \nPrecisão: {avaliacao.Precisao:P2} \nRecall: {avaliacao.Recall:P2}\nF1Score: {avaliacao.F1Score:P2}\n");
            Console.WriteLine("Melhor resultado: " + melhorResultado.TrainerName);

            _modeloTreinado = melhorResultado.Model;
        }
    }
}
