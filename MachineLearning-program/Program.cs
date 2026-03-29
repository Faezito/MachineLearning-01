using MachineLearning_01.ML;
using MachineLearning_01.Models;

void main()
{
    //ExemploClassificacaoBinaria();
    ClassificacaoMulticlasse();
}

main();



void ClassificacaoMulticlasse()
{
    var trainer = new PerfilAlunoModelTrainer();
    var caminhoCSV = Path.Combine(AppContext.BaseDirectory, "perfil_aluno_idiomas.csv");
    var caminhoModelo = Path.Combine(AppContext.BaseDirectory, "modelo_classificacao_multiclasse.zip");


    trainer.CarregarDadosCSV(caminhoCSV);
    trainer.TreinarModelo();
    trainer.AvaliarModelo();
    trainer.AvaliarMelhorModelo();
    trainer.SalvarModelo(caminhoModelo);

    var predictor = new PerfilAlunoModelPredictor();
    predictor.CarregarModelo(caminhoModelo);

    PerfilAlunoInputDataModel novoAluno = new()
    {
        NotaProficienciaGramatical = 6.5f,
        NotaCompreensaoOral = 7.0f,
        NotaConversacao = 5.5f
    };

    var res = predictor.Prever(novoAluno);
    Console.WriteLine($"Perfil previsto: {res.PerfilPrevisto}");

    Console.WriteLine("Pontuação por perfil: ");
    var perfis = new[]
    {
        "Iniciante",
        "Intermediario",
        "Avançado"
    };

    for( int i = 0; i < res.Score.Length; i++)
    {
        Console.WriteLine($"{perfis[i]}: {res.Score[i]:P2}");
    }
};






void AvaliarCasa()
{
    Console.WriteLine("Digite o tamanho da casa: ");
    float tamanho = float.Parse(Console.ReadLine());

    Console.WriteLine("Quantos quartos?");
    float quartos = float.Parse(Console.ReadLine());

    CasaInputData casaNova = new()
    {
        Tamanho = tamanho,
        Quartos = quartos
    };

    ExemploRegressao(casaNova);
}

void ExemploRegressao(CasaInputData casaNova)
{
    CasaModelTrainer trainer = new();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento_grande.csv"));

    trainer.TreinarModelo();
    trainer.AvaliarModelo();
    //trainer.AvaliarMelhorAlgoritmo();

    var caminhoModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_regressao.zip");
    trainer.SalvarModelo(caminhoModelo);

    CasaModelPredictor predictor = new();
    predictor.CarregarModelo(caminhoModelo);

    var res = predictor.Prever(casaNova);

    Console.WriteLine("Valor: " + res.PrecoPrevisto);
}


void ExemploClassificacaoBinaria()
{
    var caminhoModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_classificacao_binaria.zip");
    var caminhoDados = Path.Combine(AppContext.BaseDirectory, "aprovacao_credito.csv");

    // Treinar modelo de ML (só usar até estar treinado e encontrar um bom modelo

    //CreditoModelTrainer trainer = new();

    //trainer.CarregarDadosCSV(caminhoDados);
    //trainer.TreinarModelo();
    //trainer.AvaliarModelo();
    //trainer.EncontrarMelhorAlgoritmo();

    //trainer.SalvarModelo(caminhoModelo);


    // Usar modelo de ML
    CreditoModelPredictor predictor = new();
    predictor.CarregarModelo(caminhoModelo);

    CreditoInputDataModel creditoNovo = new()
    {
        RendaMensal = 4200f,
        EstadoCivil = 3,
        NumeroDependentes = 5,
        PossuiVeiculo = 1,
        JaNegadoAntes = 0
    };

    var res = predictor.Prever(creditoNovo);
    Console.WriteLine($"Aprovado? {(res.PredicaoAprovado ? "Sim" : "Não")}");
    Console.WriteLine($"Probabilidade: {res.Probability:P2}");
}