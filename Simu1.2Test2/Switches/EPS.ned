

package Switches;


module EPS
{
    parameters:

 //       int epsid;//electronic switch address      
        int ocsnum=2;
        int nodenum=10;
        int neighbor=2;

        @display("bgb=279,202");
        @unit;
    gates:
        inout port[nodenum+neighbor+ocsnum];
//        inout optport[];//光电转换？光端口

//        input localIn[];//nodenum
//        output localOut[];//nodenum

        input routingctrlin;
        output routingctrlout;
    submodules:
        routing: Routing {
            @display("p=160.36667,129.23666");
        }
        queue[neighbor+ocsnum]: Queue {
            @display("p=72.63667,66.03333");
        }

    connections:
        //        for i=0..neighbor-1 {
        //            routing.out[i] --> queueE[i].in;
        //            routing.in[i] <-- queueE[i].out;
        //            queueE[i].line <--> port[i];
        //        }
        //        for i=0..ocsnum-1 {//optical
        //            routing.optout[i] --> queueO[i].in;
        //            routing.optin[i] <-- queueO[i].out;
        //            queueO[i].line <--> optport[i];
        //        }        
        for i=0..nodenum-1 {
            routing.out[i] --> port$o[i];
            routing.in[i] <-- port$i[i];
        }
        for i=nodenum..nodenum+ocsnum+neighbor-1{
            routing.out[i] --> queue[i-nodenum].in;
            routing.in[i] <-- queue[i-nodenum].out;
        }
        for i=nodenum..nodenum+ocsnum+neighbor-1 {
            queue[i-nodenum].line <--> port[i];
        }

        routing.ctrlout --> routingctrlout;
        routingctrlin --> routing.ctrlin;
}

simple Queue
{
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
