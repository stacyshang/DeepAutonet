
package ElecTree;
//import Switches.*;

module CoreRouter
{
    parameters:
        int fournum = 4;
        int type = 3;
        @unit;
    gates:
        inout port[fournum];

    submodules:
        equeue[fournum]: eQueue {

            @display("p=188,39");
        }
        fourRouting: FourPortRouting {

            @display("p=38,39");
        }

    connections:
        for i=0..fournum-1 {
            fourRouting.in[i] <-- equeue[i].out;
            fourRouting.out[i] --> equeue[i].in;

            equeue[i].line <--> port[i];
        }

}

simple FourPortRouting
{
    parameters:

    gates:
        input in[4];
        output out[4];


}
