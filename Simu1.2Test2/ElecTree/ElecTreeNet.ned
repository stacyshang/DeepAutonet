
package ElecTree;
//channel
channel EPSdelay extends ned.DatarateChannel
{
    delay = 400 ns;
    datarate = default(1Gbps);
}
channel EPSdelayL2 extends ned.DatarateChannel
{
    delay = 400 ns;
    datarate = default(2Gbps);
}
channel halfEPSdelay extends ned.DatarateChannel
{
    delay = 200 ns;
    //datarate = default(1Gbps);
}

network ElecTreeNet
{
    parameters:
        int NumNodes = 160;
        int racknum = 16;

        @display("bgb=288,167");
    gates:

    submodules:
        erack[racknum]: eRack {
            erackid = index;
            @display("p=49,134");
        }

        layerRouter[4]: LayerRouter {
            lrid = index;
            @display("p=120,79");
        }
        coreRouter: CoreRouter {

            @display("p=189,26");
        }
    connections:
        erack[0].o --> EPSdelay --> layerRouter[0].port$i[0];
        erack[1].o --> EPSdelay --> layerRouter[0].port$i[1];
        erack[2].o --> EPSdelay --> layerRouter[0].port$i[2];
        erack[3].o --> EPSdelay --> layerRouter[0].port$i[3];
        erack[4].o --> EPSdelay --> layerRouter[1].port$i[0];
        erack[5].o --> EPSdelay --> layerRouter[1].port$i[1];
        erack[6].o --> EPSdelay --> layerRouter[1].port$i[2];
        erack[7].o --> EPSdelay --> layerRouter[1].port$i[3];
        erack[8].o --> EPSdelay --> layerRouter[2].port$i[0];
        erack[9].o --> EPSdelay --> layerRouter[2].port$i[1];
        erack[10].o --> EPSdelay --> layerRouter[2].port$i[2];
        erack[11].o --> EPSdelay --> layerRouter[2].port$i[3];
        erack[12].o --> EPSdelay --> layerRouter[3].port$i[0];
        erack[13].o --> EPSdelay --> layerRouter[3].port$i[1];
        erack[14].o --> EPSdelay --> layerRouter[3].port$i[2];
        erack[15].o --> EPSdelay --> layerRouter[3].port$i[3];

        erack[0].i <-- EPSdelay <-- layerRouter[0].port$o[0];
        erack[1].i <-- EPSdelay <-- layerRouter[0].port$o[1];
        erack[2].i <-- EPSdelay <-- layerRouter[0].port$o[2];
        erack[3].i <-- EPSdelay <-- layerRouter[0].port$o[3];
        erack[4].i <-- EPSdelay <-- layerRouter[1].port$o[0];
        erack[5].i <-- EPSdelay <-- layerRouter[1].port$o[1];
        erack[6].i <-- EPSdelay <-- layerRouter[1].port$o[2];
        erack[7].i <-- EPSdelay <-- layerRouter[1].port$o[3];
        erack[8].i <-- EPSdelay <-- layerRouter[2].port$o[0];
        erack[9].i <-- EPSdelay <-- layerRouter[2].port$o[1];
        erack[10].i <-- EPSdelay <-- layerRouter[2].port$o[2];
        erack[11].i <-- EPSdelay <-- layerRouter[2].port$o[3];
        erack[12].i <-- EPSdelay <-- layerRouter[3].port$o[0];
        erack[13].i <-- EPSdelay <-- layerRouter[3].port$o[1];
        erack[14].i <-- EPSdelay <-- layerRouter[3].port$o[2];
        erack[15].i <-- EPSdelay <-- layerRouter[3].port$o[3];

        for i=0..3 {
            layerRouter[i].port[4] <--> EPSdelayL2 <--> coreRouter.port[i];
        }
}

module eRack
{
    parameters:
        int erackid;
        int noodenum = 10;

        @display("bgb=208.01001,164.70001");
    gates:
        input i;//only one connecting to one CoreRouter
        output o;

    submodules:

        enode[noodenum]: eNode {
            id = index;
            @display("p=56,118");
        }

        eeps: eEPS {

            @display("p=137,51");
        }
    connections:
        for i=0..noodenum-1 {
            eeps.out[i] --> halfEPSdelay --> enode[i].appin;
            eeps.in[i] <-- halfEPSdelay <-- enode[i].appout; 
        }

        eeps.in[noodenum] <-- i;
        eeps.out[noodenum] --> o;

}

module eNode
{
    parameters:
        int id;  //Address
        int type = 0;
        @unit;
        @display("bgb=113,84");
    gates:
        input appin;
        output appout;

    submodules:
        eapp: eApp {

            @display("p=55,38");
        }

    connections:
        appin --> eapp.in;
        appout <-- eapp.out;
}

simple eApp
{
    parameters:
        volatile double pattern;
        volatile double sendIaTime @unit(s);
        volatile int packetLength @unit(byte);
    gates:
        input in;
        output out;
}

module eEPS
{
    parameters:
        int type = 1;
        int noodenum = 10;
        @unit;
        @display("bgb=192,180");
    gates:
        input in[noodenum+1];//
        output out[noodenum+1];

    submodules:
        erouting: eRouting {

            @display("p=98,123");
        }

        equeue: eQueue {

            @display("p=98,45");
        }
    connections:
        for i=0..noodenum-1 {
            erouting.in[i] <-- in[i];
            erouting.out[i] --> out[i];
        }

        erouting.out[noodenum] --> equeue.in;
        erouting.in[noodenum] <-- equeue.out;

        equeue.line$i <-- in[noodenum];
        equeue.line$o --> out[noodenum];
}
simple eQueue{
        parameters:
        int frameCapacity ; // max number of packets; 0 means no limit
        bool useCutThroughSwitching = default(false);  // use cut-through switching instead of store-and-forward
        //@display("i=block/queue;q=queue");
//        @signal[qlen](type="long");
//        @signal[busy](type="bool");
//        @signal[queueingTime](type="simtime_t");
//        @signal[drop](type="long");
//        @signal[txBytes](type="long");
//        @signal[rxBytes](type="long");
//        @statistic[qlen](title="queue length";record=vector?,timeavg,max;interpolationmode=sample-hold);
//        @statistic[busy](title="server busy state";record=vector?,timeavg;interpolationmode=sample-hold);
//        @statistic[queueingTime](title="queueing time at dequeue";unit=s;interpolationmode=none);
//        @statistic[drop](title="dropped packet byte length";unit=bytes;record=vector?,count,sum;interpolationmode=none);
//        @statistic[txBytes](title="transmitting packet byte length";unit=bytes;record=vector?,count,sum,histogram;interpolationmode=none);
//        @statistic[rxBytes](title="received packet byte length";unit=bytes;record=vector?,count,sum,histogram;interpolationmode=none);

    gates:
        input in;
        output out;
        inout line;
}

simple eRouting
{
    parameters:

    gates:
        input in[11];
        output out[11];


}

