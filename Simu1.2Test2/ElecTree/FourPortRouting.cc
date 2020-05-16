
#include <map>
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"

using namespace omnetpp;

class FourPortRouting : public cSimpleModule{
    private:
       int myAddress;

       typedef std::map<int,int> RoutingTable; // destaddr -> gateindex
       RoutingTable rtable;

       int Nodenum = 10;

    protected:
       virtual void initialize();
       virtual void handleMessage(cMessage *msg);
};
Define_Module(FourPortRouting);

void FourPortRouting::initialize(){
    myAddress = 11111;

    //initially establish routingtable when not receiving OCS connecting
    cTopology *topo = new cTopology("topo");
    topo->extractByProperty("unit");//Node+eEPS+LR+CR
    cTopology ::Node *thisNode = topo->getNodeFor(getParentModule());
    EV << "cTopology found" << topo->getNumNodes() << " nodes\n";

        // find and store next hops
        for (int i = 0; i < topo->getNumNodes(); i++) {
            if (topo->getNode(i) == thisNode)
                continue;  // skip ourselves
            topo->calculateUnweightedSingleShortestPathsTo(topo->getNode(i));

            if (thisNode->getNumPaths() == 0)
                continue;  // not connected

            cGate *parentModuleGate = thisNode->getPath(0)->getLocalGate();
            int gateIndex = parentModuleGate->getIndex();
            int address;
            //if it's Node
            if((int)topo->getNode(i)->getModule()->par("type") == 0){//不知道有没有问题！！
                address= 10*(int)topo->getNode(i)->getModule()->getParentModule()->par("erackid")+(int)topo->getNode(i)->getModule()->par("id");
            }   //is eEPS
            else if((int)topo->getNode(i)->getModule()->par("type") == 1){
                address = 1000+(int)topo->getNode(i)->getModule()->getParentModule()->par("erackid");
            }//is LayerRouter
            else if((int)topo->getNode(i)->getModule()->par("type") == 2){
                address = 10000+(int)topo->getNode(i)->getModule()->par("lrid");
            }//is CoreRouter
            else if((int)topo->getNode(i)->getModule()->par("type") == 3){
                address = 11111;
            }
            else{
                address = -1;
                EV << " node module type is illegal" << endl;
            }
            rtable[address] = gateIndex;
            //EV << "  towards address " << address << " gateIndex is " << gateIndex << endl;
        }
        delete topo;

}
void FourPortRouting::handleMessage(cMessage *msg){

    //packet
    if(dynamic_cast <Packet *> (msg) != NULL){

         Packet *pk = check_and_cast<Packet *>(msg);
         int destAddr = pk->getDestAddr();
         //int srcAddr = pk->getSrcAddr();

         //dest is a routing module (which is impossible)
         if (destAddr == myAddress) {
             EV << "discarding illegal local routing delivery of packet " << pk->getName() << endl;
             delete pk;
             return;
         }

         RoutingTable::iterator it = rtable.find(destAddr);
         if (it == rtable.end()) {
             EV << "address " << destAddr << " unreachable, discarding packet " << pk->getName() << endl;
             delete pk;
             return;
         }
         int outGateIndex = (*it).second;
         EV << "forwarding packet " << pk->getName() << " on gate index " << outGateIndex << endl;
         pk->setHopCount(pk->getHopCount()+1);
         send(pk, "out", outGateIndex);
    }

}

