
package ElecTree;


module LayerRouter
{
    parameters:
        int lrid;
        int fivenum = 5;
        int type = 2;
        @unit;
    gates:
        inout port[fivenum];

    submodules:
        equeue[fivenum]: eQueue {

            @display("p=188,39");
        }
        fiveRouting: FivePortRouting {

            @display("p=38,39");
        }

    connections:
        for i=0..fivenum-1 {
            fiveRouting.in[i] <-- equeue[i].out;
            fiveRouting.out[i] --> equeue[i].in;

            equeue[i].line <--> port[i];
        }

}

simple FivePortRouting
{
    parameters:

    gates:
        input in[5];
        output out[5];


}

