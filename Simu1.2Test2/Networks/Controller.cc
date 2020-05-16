//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

//#include "Controller.h"
#include <map>
//#include "..\Switches\Routing.h"
//#include "..\Nodes\App.h"
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"
#include "..\Messages\CtrltoOCSTopoAdd_m.h"
#include "..\Messages\CtrltoOCSTopoCut_m.h"
#include "..\Messages\CtrltoRoutingTableMod_m.h"
#include "string.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include "Statistic.h"

using namespace omnetpp;
//using namespace std;

class Controller : public cSimpleModule{
    private:
 //      int myAddress;
       typedef std::map<int,int> RoutingTable; // destaddr -> gateindex
       RoutingTable rtable;

       std::map<int,int> ocs0map,ocs1map;
       std::vector<int> a,b;

       int mtxflag;
       std::vector< vector <double> >  Matrix;
       double simulationDuration;
       double MAXSIM;
       string filename;

       cPar *tp;//topo type
       cMessage* generateNothing;
       cMessage* generateMesh;
       cMessage* generateTorus;
       cMessage* generateOptimal;
       cMessage* generateDragonfly;//self message
       //cMessage* change0t1;

       cMessage* checkMatrix;
       cMessage* printNow;
       simtime_t flagStartTime;

    protected:
       virtual void initialize();
       virtual void handleMessage(cMessage *msg);
       virtual void calculateNewWeightedSingleShortestPathsTo(cTopology ::Node *_target);
};
Define_Module(Controller);

void Controller::initialize(){
 //   myAddress = par("address");
    tp = &par("tp");
    generateNothing = new cMessage("nothing");
    generateMesh = new cMessage("nextMesh");
    generateTorus = new cMessage("nextTotus");
    generateOptimal = new cMessage("nextOptimal");
    generateDragonfly = new cMessage("nextDragonfly");
    if(tp->doubleValue() == (double)1){  //Ring
        scheduleAt(0.0,generateNothing);
    }
    else if(tp->doubleValue() == (double)2){
        scheduleAt(0.0,generateMesh);
        EV<<"generateMesh = new cMessage"<<endl;
    }
    else if(tp->doubleValue() == (double)3){
        scheduleAt(0.0,generateTorus);
    }
    else if(tp->doubleValue() == (double)4){
        scheduleAt(0.0,generateOptimal);
    }
    else if(tp->doubleValue() == (double)5){
        scheduleAt(0.0,generateDragonfly);
    }
    //continuously check matrix change
    mtxflag = 0;
    checkMatrix = new cMessage("checkMatrixChange");
    scheduleAt(0.0, checkMatrix);
    EV<<"checkMatrix now for the 1st time"<<endl;
    Matrix = std::vector<vector<double> > (16, vector<double>(16));
    for (int src = 0; src<16; src++){
        for (int dst = 0; dst<16; dst++){
            (Matrix)[src][dst] = 0;
        }
    }
    for (int ii=0; ii<=14; ii++){
        (Matrix)[ii][ii+1] = 1;
    }(Matrix)[15][0] = 1;
    for (int jj=1; jj<=15; jj++){
        (Matrix)[jj][jj-1] = 1;
    }(Matrix)[0][15] = 1;

    ofstream myfile;
    filename = "NetworkMatrix.txt";
    myfile.open (filename, ios::out | ios::trunc );
    myfile << mtxflag <<"\n";
    for (int src = 0; src<16; src++){
        for (int dst = 0; dst<16; dst++){
            int d = Matrix[src][dst];
            myfile  << d;
            if (dst != 15){myfile << ",";}
        }
        myfile << "\n";
    }
    myfile << endl;
    myfile.close();

    printNow = new cMessage("printStatusNow");
    scheduleAt(0.0, printNow);
//    startTime = par("startTime");
    MAXSIM = par("simulationDuration");
    Statistic::instance()->setMaxSim(MAXSIM);
}
void Controller::handleMessage(cMessage *msg){

    if(msg == generateNothing){
        //£¡£¡
        EV << "Receive generateNothing " << msg->getName() << endl;
    }
    else if(msg == printNow){

        Statistic::instance()->infoTS(simTime());
        //Statistic::instance()->printStats();
        scheduleAt(simTime()+0.00001, printNow);
    }
    else if(msg == checkMatrix){
        //Statistic::instance()->infoTS(simTime());
        ifstream myfile;
        myfile.open (filename, ios::in );
        string line ;
        if (myfile.is_open()){
            while ( getline (myfile,line) ){
               //EV << line << endl;
               break;//just get the first line
            }
            myfile.close();
        }
        else{
            EV << "Unable to open file";}
        int flag = std::stoi(line);
        //EV <<"line = "<< flag << endl;
        if(flag != mtxflag){
            //flag from tensorflow changes, need to change topo, calculate mean E2E delay and print it into Delay.txt
            if(flag == 2||flag == 22){scheduleAt(simTime(), generateMesh);}
            if(flag == 3||flag == 33){scheduleAt(simTime(), generateTorus);}
            if(flag == 4||flag == 44){scheduleAt(simTime(), generateOptimal);}
            if(flag == 5||flag == 55){scheduleAt(simTime(), generateDragonfly);}
            flagStartTime = simTime();
            Statistic::instance()->INI = flagStartTime + 0.001;// for warm-up
            Statistic::instance()->END = flagStartTime + 0.001 + 0.005;//5ms
            EV<<"printNow, INI="<< Statistic::instance()->INI <<", END="<<Statistic::instance()->END<<endl;
            //Statistic::instance()->collect = true;
            //scheduleAt(flagStartTime + 1, printNow);//run for 4s, and TF read 5s later
            mtxflag = flag;
        }
//        if (simTime() < MAXSIM) {
           scheduleAt(simTime() + 0.00001, checkMatrix);//loop detection
//        }
//        else {
//           EV << "END simulation" << endl;
//        }
    }
    else if(msg == generateMesh || msg == generateTorus || msg == generateOptimal || msg == generateDragonfly){
        //RESET OCS connections************
        if(ocs0map.size()!=0 ){
            std::map<int,int>::iterator i;
            CtrltoOCSTopoCut* tm[ocs0map.size()];
            for(i=ocs0map.begin(); i!=ocs0map.end(); i++){
                int x = 0;
                tm[x] = new CtrltoOCSTopoCut;tm[x]->setInGate(i->first);tm[x]->setOutGate(i->second);
                send(tm[x], "ocsout", 0);
                x++;
            }
        }
        if(ocs1map.size()!=0){
            CtrltoOCSTopoCut* tn[ocs1map.size()];
            std::map<int,int>::iterator j;
            for(j=ocs1map.begin(); j!=ocs1map.end(); j++){
                int y = 0;
                tn[y] = new CtrltoOCSTopoCut;tn[y]->setInGate(j->first);tn[y]->setOutGate(j->second);
                send(tn[y], "ocsout", 1);
                y++;
            }
        }
        EV << "sending ctrltoOCSTopo Cut msg"<< endl;
        ocs0map.clear();
        ocs1map.clear();
        Matrix.clear();
        for (int src = 0; src<16; src++){
            for (int dst = 0; dst<16; dst++){(Matrix)[src][dst] = 0;}
        }
        for (int ii=0; ii<=14; ii++){(Matrix)[ii][ii+1] = 1; }(Matrix)[15][0] = 1;
        for (int jj=1; jj<=15; jj++){(Matrix)[jj][jj-1] = 1; }(Matrix)[0][15] = 1;
        //***************
        if (msg == generateMesh) {
            mtxflag = 2;
            ocs0map.insert(std::map<int,int>::value_type(3,6));Matrix[3][6]=1;
            ocs0map.insert(std::map<int,int>::value_type(6,3));Matrix[6][3]=1;
            ocs0map.insert(std::map<int,int>::value_type(10,7));Matrix[10][7]=1;
            ocs0map.insert(std::map<int,int>::value_type(7,10));Matrix[7][10]=1;
            ocs0map.insert(std::map<int,int>::value_type(11,14));Matrix[11][14]=1;
            ocs0map.insert(std::map<int,int>::value_type(14,11));Matrix[14][11]=1;
            ocs0map.insert(std::map<int,int>::value_type(2,15));Matrix[2][15]=1;
            ocs0map.insert(std::map<int,int>::value_type(15,2));Matrix[15][2]=1;
            ocs1map.insert(std::map<int,int>::value_type(1,4));Matrix[1][4]=1;
            ocs1map.insert(std::map<int,int>::value_type(4,1));Matrix[4][1]=1;
            ocs1map.insert(std::map<int,int>::value_type(2,11));Matrix[2][11]=1;
            ocs1map.insert(std::map<int,int>::value_type(11,2));Matrix[11][2]=1;
            ocs1map.insert(std::map<int,int>::value_type(12,9));Matrix[12][9]=1;
            ocs1map.insert(std::map<int,int>::value_type(9,12));Matrix[9][12]=1;
            ocs1map.insert(std::map<int,int>::value_type(3,10));Matrix[3][10]=1;
            ocs1map.insert(std::map<int,int>::value_type(10,3));Matrix[10][3]=1;

            //ocs0map={{3,6},{6,3},{10,7},{7,10},{11,14},{14,11},{2,15},{15,2}};
            //ocs1map={{1,4},{4,1},{2,11},{11,2},{12,9},{9,12},{3,10},{10,3}};
        }
        else if (msg == generateTorus){//Torus16
            mtxflag = 3;
            ocs0map.insert(std::map<int,int>::value_type(3,6));Matrix[3][6]=1;
            ocs0map.insert(std::map<int,int>::value_type(6,3));Matrix[3][6]=1;
            ocs0map.insert(std::map<int,int>::value_type(10,7));Matrix[10][7]=1;
            ocs0map.insert(std::map<int,int>::value_type(7,10));Matrix[7][10]=1;
            ocs0map.insert(std::map<int,int>::value_type(11,14));Matrix[11][14]=1;
            ocs0map.insert(std::map<int,int>::value_type(14,11));Matrix[14][11]=1;
            ocs0map.insert(std::map<int,int>::value_type(2,15));Matrix[2][15]=1;
            ocs0map.insert(std::map<int,int>::value_type(15,2));Matrix[15][2]=1;
            ocs0map.insert(std::map<int,int>::value_type(0,5));Matrix[0][5]=1;
            ocs0map.insert(std::map<int,int>::value_type(5,0));Matrix[5][0]=1;
            ocs0map.insert(std::map<int,int>::value_type(8,13));Matrix[8][13]=1;
            ocs0map.insert(std::map<int,int>::value_type(13,8));Matrix[13][8]=1;
            ocs0map.insert(std::map<int,int>::value_type(1,12));Matrix[1][12]=1;
            ocs0map.insert(std::map<int,int>::value_type(12,1));Matrix[12][1]=1;
            ocs0map.insert(std::map<int,int>::value_type(4,9));Matrix[4][9]=1;
            ocs0map.insert(std::map<int,int>::value_type(9,4));Matrix[9][4]=1;

            ocs1map.insert(std::map<int,int>::value_type(1,4));Matrix[1][4]=1;
            ocs1map.insert(std::map<int,int>::value_type(4,1));Matrix[4][1]=1;
            ocs1map.insert(std::map<int,int>::value_type(2,11));Matrix[2][11]=1;
            ocs1map.insert(std::map<int,int>::value_type(11,2));Matrix[11][2]=1;
            ocs1map.insert(std::map<int,int>::value_type(12,9));Matrix[12][9]=1;
            ocs1map.insert(std::map<int,int>::value_type(9,12));Matrix[9][12]=1;
            ocs1map.insert(std::map<int,int>::value_type(3,10));Matrix[3][10]=1;
            ocs1map.insert(std::map<int,int>::value_type(10,3));Matrix[10][3]=1;
            ocs1map.insert(std::map<int,int>::value_type(6,15));Matrix[6][15]=1;
            ocs1map.insert(std::map<int,int>::value_type(15,6));Matrix[15][6]=1;
            ocs1map.insert(std::map<int,int>::value_type(7,14));Matrix[7][14]=1;
            ocs1map.insert(std::map<int,int>::value_type(14,7));Matrix[14][7]=1;
            ocs1map.insert(std::map<int,int>::value_type(0,13));Matrix[0][13]=1;
            ocs1map.insert(std::map<int,int>::value_type(13,0));Matrix[13][0]=1;
            ocs1map.insert(std::map<int,int>::value_type(5,8));Matrix[5][8]=1;
            ocs1map.insert(std::map<int,int>::value_type(8,5));Matrix[8][5]=1;
            //ocs0map={{3,6},{6,3},{10,7},{7,10},{11,14},{14,11},{2,15},{15,2},{0,5},{5,0},{8,13},{13,8},{1,12},{12,1},{4,9},{9,4}};
            //ocs1map={{1,4},{4,1},{2,11},{11,2},{12,9},{9,12},{3,10},{10,3},{6,15},{15,6},{7,14},{14,7},{0,13},{13,0},{5,8},{8,5}};
        }
        else if(msg == generateOptimal){//Optimal N16k3
            mtxflag = 4;
            ocs0map.insert(std::map<int,int>::value_type(0,8));Matrix[0][8]=1;
            ocs0map.insert(std::map<int,int>::value_type(8,0));Matrix[8][0]=1;
            ocs0map.insert(std::map<int,int>::value_type(1,11));Matrix[1][11]=1;
            ocs0map.insert(std::map<int,int>::value_type(11,1));Matrix[11][1]=1;
            ocs0map.insert(std::map<int,int>::value_type(2,7));Matrix[2][7]=1;
            ocs0map.insert(std::map<int,int>::value_type(7,2));Matrix[7][2]=1;
            ocs0map.insert(std::map<int,int>::value_type(3,13));Matrix[3][13]=1;
            ocs0map.insert(std::map<int,int>::value_type(13,3));Matrix[13][3]=1;
            ocs1map.insert(std::map<int,int>::value_type(4,10));Matrix[4][10]=1;
            ocs1map.insert(std::map<int,int>::value_type(10,4));Matrix[10][4]=1;
            ocs1map.insert(std::map<int,int>::value_type(5,15));Matrix[5][15]=1;
            ocs1map.insert(std::map<int,int>::value_type(15,5));Matrix[15][5]=1;
            ocs1map.insert(std::map<int,int>::value_type(6,12));Matrix[6][12]=1;
            ocs1map.insert(std::map<int,int>::value_type(12,6));Matrix[12][6]=1;
            ocs1map.insert(std::map<int,int>::value_type(9,14));Matrix[9][14]=1;
            ocs1map.insert(std::map<int,int>::value_type(14,9));Matrix[14][9]=1;
            //ocs0map={{0,8},{8,0},{1,11},{11,1},{2,7},{7,2},{3,13},{13,3}};
            //ocs1map={{4,10},{10,4},{5,15},{15,5},{6,12},{12,6},{9,14},{14,9}};
        }
        else if(msg == generateDragonfly){//Dragonfly
            mtxflag = 5;
            ocs0map.insert(std::map<int,int>::value_type(0,2));Matrix[0][2]=1;
            ocs0map.insert(std::map<int,int>::value_type(2,0));Matrix[2][0]=1;
            ocs0map.insert(std::map<int,int>::value_type(1,15));Matrix[1][15]=1;
            ocs0map.insert(std::map<int,int>::value_type(15,1));Matrix[15][1]=1;
            ocs0map.insert(std::map<int,int>::value_type(3,5));Matrix[3][5]=1;
            ocs0map.insert(std::map<int,int>::value_type(5,3));Matrix[5][3]=1;
            ocs0map.insert(std::map<int,int>::value_type(4,6));Matrix[4][6]=1;
            ocs0map.insert(std::map<int,int>::value_type(6,4));Matrix[6][4]=1;
            ocs0map.insert(std::map<int,int>::value_type(7,9));Matrix[7][9]=1;
            ocs0map.insert(std::map<int,int>::value_type(9,7));Matrix[9][7]=1;
            ocs0map.insert(std::map<int,int>::value_type(8,10));Matrix[8][10]=1;
            ocs0map.insert(std::map<int,int>::value_type(10,8));Matrix[10][8]=1;
            ocs0map.insert(std::map<int,int>::value_type(11,13));Matrix[11][13]=1;
            ocs0map.insert(std::map<int,int>::value_type(13,11));Matrix[13][11]=1;
            ocs0map.insert(std::map<int,int>::value_type(12,14));Matrix[12][14]=1;
            ocs0map.insert(std::map<int,int>::value_type(14,12));Matrix[14][12]=1;

            ocs1map.insert(std::map<int,int>::value_type(2,15));Matrix[2][15]=1;
            ocs1map.insert(std::map<int,int>::value_type(15,2));Matrix[15][2]=1;
            ocs1map.insert(std::map<int,int>::value_type(3,6));Matrix[3][6]=1;
            ocs1map.insert(std::map<int,int>::value_type(6,3));Matrix[6][3]=1;
            ocs1map.insert(std::map<int,int>::value_type(7,10));Matrix[7][10]=1;
            ocs1map.insert(std::map<int,int>::value_type(10,7));Matrix[10][7]=1;
            ocs1map.insert(std::map<int,int>::value_type(11,14));Matrix[11][14]=1;
            ocs1map.insert(std::map<int,int>::value_type(14,11));Matrix[14][11]=1;
            ocs1map.insert(std::map<int,int>::value_type(1,9));Matrix[1][9]=1;
            ocs1map.insert(std::map<int,int>::value_type(9,1));Matrix[9][1]=1;
            ocs1map.insert(std::map<int,int>::value_type(4,12));Matrix[4][12]=1;
            ocs1map.insert(std::map<int,int>::value_type(12,4));Matrix[12][4]=1;

        }
        else{
            throw cRuntimeError("generateTopo msg is wrong",msg);
        }
        EV << "Receive generate Message " << msg->getName() <<", ocs0map size" << ocs0map.size()<<", ocs1map size" << ocs1map.size()<< endl;
        //update NetworkMatrix.txt
        ofstream myfile;
        filename = "NetworkMatrix.txt";
        myfile.open (filename, ios::out | ios::trunc );
        myfile << mtxflag <<"\n";
        for (int src = 0; src<16; src++){
            for (int dst = 0; dst<16; dst++){
                int d = Matrix[src][dst];
                myfile  << d;
                if (dst != 15){myfile<< ",";}
            }
            myfile << "\n";
        }
        myfile << endl;
        myfile.close();

        //to set OCS connection
        std::map<int,int>::iterator i;
        std::map<int,int>::iterator j;
        CtrltoOCSTopoAdd* tm[ocs0map.size()];
        for(i=ocs0map.begin(); i!=ocs0map.end(); i++){
            int x = 0;
            tm[x] = new CtrltoOCSTopoAdd;tm[x]->setInGate(i->first);tm[x]->setOutGate(i->second);
            send(tm[x], "ocsout", 0);
            x++;
        }

        CtrltoOCSTopoAdd* tn[ocs1map.size()];
        for(j=ocs1map.begin(); j!=ocs1map.end(); j++){
            int y = 0;
            tn[y] = new CtrltoOCSTopoAdd;tn[y]->setInGate(j->first);tn[y]->setOutGate(j->second);
            send(tn[y], "ocsout", 1);
            y++;
        }
        EV << "sending ctrltoOCSTopoAdd & Cut msg"<< endl;

        //send ocsmap to Routing for routing table calculate
        std::map<int,int>::iterator it,itt;
        CtrltoRoutingTableMod *mod = new CtrltoRoutingTableMod;
        mod->setOcs0mapsize(ocs0map.size());mod->setOcs1mapsize(ocs1map.size());
        EV<<"mod->getOcs0mapsize: "<<ocs0map.size()<<",mod->setOcs1mapsize: "<<ocs1map.size()<<endl;
        int x = 0;int y = 0;
        for(it=ocs0map.begin(); it!=ocs0map.end(); it++){
            mod->setOcs0mapleft(x, it->first);
            mod->setOcs0mapright(x, it->second);
            x++;
        }
        for(itt=ocs1map.begin(); itt!=ocs1map.end(); itt++){
            mod->setOcs1mapleft(y, itt->first);
            mod->setOcs1mapright(y, itt->second);
            y++;
        }
        CtrltoRoutingTableMod *modclone[16];
        for(int p = 0;p<16;p++){//send to 16 racks' routing
            modclone[p] = new CtrltoRoutingTableMod;
            modclone[p] = mod->dup();
            send(modclone[p], "routingout", p);
        }

        EV << "sending ctrltoOCSRoutingTableMod msg"<< endl;
    }
    else{   //receive Packet to be processed
        if(dynamic_cast <Packet *> (msg) != NULL){
//        Packet* s = dynamic_cast<Packet* >(msg);
//            if(s){
                Packet *pk = check_and_cast<Packet *>(msg);
                EV << "abnormally received packet " << pk->getName() << " after " << pk->getHopCount() << "hops" << endl;
                delete pk;
            }

    }

}
//new shortest path
void Controller::calculateNewWeightedSingleShortestPathsTo(cTopology ::Node *_target){
//    if (!_target)
//            throw cRuntimeError(this, "..ShortestPathTo(): Target node is nullptr");
//
//    std::vector<Node*> nodes;
//    cTopology ::Node *target = _target;
//    for (int i = 0; i < (int)nodes.size(); i++) {
//            nodes[i]->dist = INFINITY;
//            nodes[i]->outPath = nullptr;
//    }
//    target->dist = 0; //distance
//
//    std::list<cTopology ::Node *> q;
//    q.push_back(target);  //insert at the bottle
//
//    while (!q.empty()) {
//        cTopology::Node *dest = q.front();//return the first top data
//        q.pop_front();//remove the first data
//
//        // for each w adjacent to v...
//        for (int i = 0; i < dest->getNumInLinks(); i++) {
//            if (!(dest->getLinkIn(i)->isEnabled()))
//                continue;
//
//            cTopology::Node *src = dest->getLinkIn(i)->getRemoteNode();
////            //if the source module is OCS, that is complex
////            if(src->getModule() == OCS){
////
////            }
//
//            if (!src->isEnabled())
//                continue;
//
//            double linkWeight = dest->getLinkIn(i)->getWeight();
//            ASSERT(linkWeight > 0.0);
//
//            double newdist = dest->dist + linkWeight;
//            if (dest != target)
//                newdist += dest->getWeight();  // dest is not the target, uses weight of dest node as price of routing (infinity means dest node doesn't route between interfaces)
//            if (newdist != INFINITY && src->dist > newdist) {  // it's a valid shorter path from src to target node
//                if (src->dist != INFINITY)
//                    q.remove(src);  // src is in the queue
//                    src->dist = newdist;
//                    src->outPath = dest->inLinks[i];
//
//                    // insert src node to ordered list
//                    std::list<cTopology::Node *>::iterator it;
//                    for (it = q.begin(); it != q.end(); ++it)
//                        if ((*it)->dist > newdist)
//                             break;
//
//                    q.insert(it, src);
//            }
//        }
//    }
}

//Controller::Controller() {
//    // TODO Auto-generated constructor stub
//
//}
//
//Controller::~Controller() {
//    // TODO Auto-generated destructor stub
//}


