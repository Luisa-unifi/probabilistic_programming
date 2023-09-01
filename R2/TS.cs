using MicrosoftResearch.R2Lib;
using MicrosoftResearch.R2Lib.Distributions;

class TrueSkill
{
    int[] players = new int[3] { 0, 1, 2 };

    int[,] games = { { 0, 1}, { 1, 2}, { 0, 2} };

    public double GetPlayerSkills()
    {
        double[] playerSkills = new double[3];

        int i;
        for (i = 0; i < 3; ++i)
        {
            playerSkills[i] = Normal.Sample(1, 0.1);
        }

        double[,] performance = new double[3, 2];

        for (i = 0; i < 3; ++i)
        {
            performance[i, 0] = Normal.Sample(playerSkills[games[i, 0]], 0.15);
            performance[i, 1] = Normal.Sample(playerSkills[games[i, 1]], 0.15);        
            Observer.Observe(performance[i, 0] > performance[i, 1]);
           
        }

        return playerSkills[0];
    }
}
