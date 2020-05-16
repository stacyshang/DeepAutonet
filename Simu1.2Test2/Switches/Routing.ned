 
package Switches;

simple Routing
{
    parameters:
        int nodenum=10;
        int ocsnum=2;
        int neighbor=2;
      //  @display("i=block/switch");
//        @signal[drop](type="long");
//        @signal[outputIf](type="long");
//        @statistic[drop](title="dropped packet byte length";unit=bytes;record=vector?,count,sum;interpolationmode=none);
//        @statistic[outputIf](title="output interface for each routed packet";record=histogram;interpolationmode=none);
    gates:
        input in[nodenum+ocsnum+neighbor];
        output out[nodenum+ocsnum+neighbor];
//        input optin[];
//        output optout[];
//        input localIn[];//connect to Node(App)
//        output localOut[];

        input ctrlin;
        output ctrlout;
}