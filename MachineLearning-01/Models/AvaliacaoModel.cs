using System.Transactions;

namespace MachineLearning_01.Models
{
    public class AvaliacaoModel
    {
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public double R2 { get; set; }
        public double Acuracia { get; set; }
        public double Precisao { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
    }
}
