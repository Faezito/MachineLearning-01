using MachineLearning_01.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MachineLearning_01.ML
{
    public class CasaModelTrainer   // CLASSE QUE FARÁ O TREINAMENTO DO MODELO DE ML
    {
        private MLContext _mlContext = new();
        private IDataView _dados;
        private ITransformer _modeloTreinado;

        public void CarregarDadosCSV(string caminho)
        {
            _dados = _mlContext.Data.LoadFromTextFile<CasaInputData>(
                    path: caminho, // onde está o csv
                    hasHeader: true, // caso o csv tenha um header, ignora a primeira linha
                    separatorChar: ','
                );
        }

        public void TreinarModelo()
        {
            var pipeline = _mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(CasaInputData.Tamanho),
                    nameof(CasaInputData.Quartos)
                )
                //.Append(_mlContext.Regression.Trainers.Sdca(
                //        labelColumnName: "Preco", // Qual coluna o modelo ML deve aprender a prever, usando as colunas mencionadas acima
                //        maximumNumberOfIterations: 100 // Interações para prever, quanto maior melhor, porém mais lento, e se for muito grande pode confundir o model
                //    )
                .Append(_mlContext.Regression.Trainers.FastForest(
                        labelColumnName: "Preco",
                        numberOfTrees: 100, //similar o iterations, quantidade de arvores de decisão a serem utilizadas
                        numberOfLeaves: 20, // complexidade das arvores de decisão utilizadas
                        minimumExampleCountPerLeaf: 10 // controle de overfitting (quando o modelo decora dados, ao invés de aprender padrões)
                    )
                );

            _modeloTreinado = pipeline.Fit(_dados); // Passa os dados para o pipeline treinar
        }

        public void SalvarModelo(string caminho)
        {
            _mlContext.Model.Save(_modeloTreinado, _dados.Schema, caminho);
        }

        public void AvaliarModelo()
        {
            var previsoes = _modeloTreinado.Transform(_dados);

            var metricas = _mlContext.Regression.Evaluate(
                    data: previsoes,
                    labelColumnName: "Preco",
                    scoreColumnName: "Score"
                );

            AvaliacaoModel avaliacao = new()
            {
                MAE = metricas.MeanAbsoluteError, //Margem de erro do modelo ML (menor = melhor)
                RMSE = metricas.RootMeanSquaredError, //Erros muito grandes do modelo (menor = melhor)
                R2 = metricas.RSquared //O quanto o modelo consegue explicar a variação de dados (0 à 1, quanto maior, melhor)
            };

            Console.WriteLine($"MAE: {avaliacao.MAE} \nRMSE: {avaliacao.RMSE} \nR2: {avaliacao.R2}\n");           
        }

        public void AvaliarMelhorAlgoritmo()
        {
            RegressionExperimentSettings experimentSettings = new()
            {
                MaxExperimentTimeInSeconds = 60
            };

            var experiment = _mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            Console.WriteLine("Encontrando melhor algoritmo...");

            var res = experiment.Execute(_dados, labelColumnName: "Preco");
            var melhorResultado = res.BestRun;

            AvaliacaoModel avaliacao = new()
            {
                MAE = melhorResultado.ValidationMetrics.MeanAbsoluteError, //Margem de erro do modelo ML (menor = melhor)
                RMSE = melhorResultado.ValidationMetrics.RootMeanSquaredError, //Erros muito grandes do modelo (menor = melhor)
                R2 = melhorResultado.ValidationMetrics.RSquared //O quanto o modelo consegue explicar a variação de dados (0 à 1, quanto maior, melhor)
            };

            Console.WriteLine($"MAE: {avaliacao.MAE} \nRMSE: {avaliacao.RMSE} \nR2: {avaliacao.R2}");
            Console.WriteLine("Melhor resultado: " + melhorResultado.TrainerName);
        }
    }
}
