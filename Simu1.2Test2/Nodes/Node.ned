
package Nodes;
//package simu1;//.Switches
import Switches.Routing;


module Node
{
    parameters:
        int id;  //Address

        //int type = 0;
        @display("bgb=30.286667,31.57");
        @unit;
    gates:
        input appin;
        output appout;

//        input ctrlin;
//        output ctrlout;

    submodules:
        app: App {
            @display("p=16.426666,15.400001");
        }

    connections:
        appin --> app.in;
        appout <-- app.out;
//        ctrlin --> app.ctrlin;
//        ctrlout <-- app.ctrlout;

}





