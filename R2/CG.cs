using MicrosoftResearch.R2Lib;
using MicrosoftResearch.R2Lib.Distributions;

class ClickGraphModel{


    public double click(){

    double simAll = Uniform.Sample(0, 1);
    bool[] clicksA = new bool[3] { true, true, false };
    bool[] clicksB = new bool[3] { true, true, false };
    double p1=0;
    double p2=0;
    int i=0;
    bool sim, clickA, clickB =false;

    for (i = 0; i < 3; ++i)
    {    
         sim = Bernoulli.Sample( simAll );         
         if (sim)
         {
             p1 = Uniform.Sample(0, 1);
             p2 = p1;
         }else{
             p1 = Uniform.Sample(0, 1);
             p2 = Uniform.Sample(0, 1);       
         }
         clickA= Bernoulli.Sample( p1 );
         clickB= Bernoulli.Sample( p2 );
         Observer.Observe(clickA == clicksA [i]);
         Observer.Observe(clickB == clicksB [i]);    
    }
    return simAll;
    }


}
