

package Nodes;

simple App
{
    parameters:
//        int address;  // local node address
//        int destAddress;  // destination address
        volatile double pattern;
        volatile double sendIaTime @unit(s);  // time between generating packets
        volatile int packetLength @unit(byte);  // length of one message (fixed! no "volatile" modifier)
//        @display("i=block/browser");
//        @signal[endToEndDelay](type="simtime_t");
//        @signal[hopCount](type="long");
//        @signal[sourceAddress](type="long");
//        @statistic[endToEndDelay](title="end-to-end delay of arrived packets";unit=s;record=vector,mean,max;interpolationmode=none);
//        @statistic[hopCount](title="hop count of arrived packets";interpolationmode=none;record=vector?,mean,max);
//        @statistic[sourceAddress](title="source address of arrived packets";interpolationmode=none;record=vector?);
        double simulationDuration;
    gates:
        input in;
        output out;
//        input ctrlin;
//        output ctrlout;
}