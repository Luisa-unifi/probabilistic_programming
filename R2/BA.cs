using MicrosoftResearch.R2Lib;
using MicrosoftResearch.R2Lib.Distributions;

class BurglarAlarmModel
{
    public bool IsBurglary()
    {
        bool earthquake, burglary, alarm, phoneWorking, maryWakes, called;

        earthquake = Bernoulli.Sample(0.001);

        burglary = Bernoulli.Sample(0.01);

        alarm = earthquake || burglary;

        if (earthquake)
        {
            phoneWorking = Bernoulli.Sample(0.6);
        }
        else
        {
            phoneWorking = Bernoulli.Sample(0.99);
        }

        if (alarm)
        {
            if (earthquake)
            {
                maryWakes = Bernoulli.Sample(0.8);
            }
            else
            {
                maryWakes = Bernoulli.Sample(0.6);
            }
        }
        else
        {
            maryWakes = Bernoulli.Sample(0.2);
        }

        called = maryWakes && phoneWorking;

        Observer.Observe(called);

        return burglary;
    }
}