
#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"
using namespace omnetpp;

class eQueue : public cSimpleModule
{
  private:
    long frameCapacity;

    cQueue queue;
    cMessage *endTransmissionEvent;
    bool isBusy;
    int dropnum;

//    simsignal_t qlenSignal;
//    simsignal_t busySignal;
//    simsignal_t queueingTimeSignal;
//    simsignal_t dropSignal;
//    simsignal_t txBytesSignal;
//    simsignal_t rxBytesSignal;

  public:
    eQueue();
    virtual ~eQueue();

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
    //virtual void refreshDisplay() const override;
    virtual void startTransmitting(cMessage *msg);
    virtual void finish();

    cStdDev dropsum;
};

Define_Module(eQueue);

eQueue::eQueue() {
    endTransmissionEvent = nullptr;
}

eQueue::~eQueue() {
    cancelAndDelete(endTransmissionEvent);
}

void eQueue::initialize()
{
    queue.setName("queue");
    endTransmissionEvent = new cMessage("endTxEvent");
    dropnum = 0;
    dropsum.setName("dropsum");

    if (par("useCutThroughSwitching"))// use cut-through switching instead of store-and-forward
        gate("line$i")->setDeliverOnReceptionStart(true);

    frameCapacity = par("frameCapacity");

//    qlenSignal = registerSignal("qlen");
//    busySignal = registerSignal("busy");
//    queueingTimeSignal = registerSignal("queueingTime");
//    dropSignal = registerSignal("drop");
//    txBytesSignal = registerSignal("txBytes");
//    rxBytesSignal = registerSignal("rxBytes");
//
//    emit(qlenSignal, queue.getLength());
//    emit(busySignal, false);

    isBusy = false;
}

void eQueue::handleMessage(cMessage *msg)
{
    if (msg == endTransmissionEvent) {
        // Transmission finished, we can start next one.
        EV << "Transmission finished.\n";
        isBusy = false;
        if (queue.isEmpty()) {
            //emit(busySignal, false);
        }
        else {
            msg = (cMessage *)queue.pop();
            //emit(queueingTimeSignal, simTime() - msg->getTimestamp());
            //emit(qlenSignal, queue.getLength());
            startTransmitting(msg);
        }
    }
    else if (msg->arrivedOn("line$i")) {
        // pass up
        //emit(rxBytesSignal, (long)check_and_cast<cPacket *>(msg)->getByteLength());
        send(msg, "out");
    }
    else {  // arrived on gate "in"
        if(msg->arrivedOn("in")) {
        if(dynamic_cast <Packet *> (msg) != NULL){
            Packet *pk = check_and_cast<Packet *>(msg);
            if (endTransmissionEvent->isScheduled()) {
                // We are currently busy, so just queue up the packet.
                if (frameCapacity && queue.getLength() >= frameCapacity) {
                    EV << "Received " << pk << " but transmitter busy and queue full: discarding\n";
                    //emit(dropSignal, (long)check_and_cast<cPacket *>(pk)->getByteLength());
                    dropnum++;
                    dropsum.collect(dropnum);
                    delete pk;
                }
                else {
                    EV << "Received " << pk << " but transmitter busy: queueing up\n";
                    pk->setTimestamp();
                    queue.insert(pk);
                    //emit(qlenSignal, queue.getLength());
                }
            }
            else {
                // We are idle, so we can start transmitting right away.
                EV << "Received " << pk << "and it's ByteLength is : "<< pk->getByteLength()<<endl;
                //emit(queueingTimeSignal, SIMTIME_ZERO);
                startTransmitting(pk);
                //emit(busySignal, true);
            }
        }
        }
    }
}

void eQueue::startTransmitting(cMessage *msg)
{
    if(dynamic_cast <Packet *> (msg) != NULL){
         Packet *pk = check_and_cast<Packet *>(msg);

         EV << "Starting transmission of " << pk << endl;
         isBusy = true;
         int64_t numBytes = check_and_cast<cPacket *>(msg)->getByteLength();
         send(pk, "line$o");

         //emit(txBytesSignal, (long)numBytes);

         // Schedule an event for the time when last bit will leave the gate.
         simtime_t endTransmission = gate("line$o")->getTransmissionChannel()->getTransmissionFinishTime();
         scheduleAt(endTransmission, endTransmissionEvent);

         //scheduleAt(simTime(), endTransmissionEvent);
    }
}

void eQueue::finish(){
    dropsum.record();
}
