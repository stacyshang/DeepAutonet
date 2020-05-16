//
// Generated file, do not edit! Created by nedtool 5.1 from Messages/CtrltoRoutingTableMod.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wc++98-compat"
#  pragma clang diagnostic ignored "-Wunreachable-code-break"
#  pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include <iostream>
#include <sstream>
#include "CtrltoRoutingTableMod_m.h"

namespace omnetpp {

// Template pack/unpack rules. They are declared *after* a1l type-specific pack functions for multiple reasons.
// They are in the omnetpp namespace, to allow them to be found by argument-dependent lookup via the cCommBuffer argument

// Packing/unpacking an std::vector
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::vector<T,A>& v)
{
    int n = v.size();
    doParsimPacking(buffer, n);
    for (int i = 0; i < n; i++)
        doParsimPacking(buffer, v[i]);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::vector<T,A>& v)
{
    int n;
    doParsimUnpacking(buffer, n);
    v.resize(n);
    for (int i = 0; i < n; i++)
        doParsimUnpacking(buffer, v[i]);
}

// Packing/unpacking an std::list
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::list<T,A>& l)
{
    doParsimPacking(buffer, (int)l.size());
    for (typename std::list<T,A>::const_iterator it = l.begin(); it != l.end(); ++it)
        doParsimPacking(buffer, (T&)*it);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::list<T,A>& l)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        l.push_back(T());
        doParsimUnpacking(buffer, l.back());
    }
}

// Packing/unpacking an std::set
template<typename T, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::set<T,Tr,A>& s)
{
    doParsimPacking(buffer, (int)s.size());
    for (typename std::set<T,Tr,A>::const_iterator it = s.begin(); it != s.end(); ++it)
        doParsimPacking(buffer, *it);
}

template<typename T, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::set<T,Tr,A>& s)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        T x;
        doParsimUnpacking(buffer, x);
        s.insert(x);
    }
}

// Packing/unpacking an std::map
template<typename K, typename V, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::map<K,V,Tr,A>& m)
{
    doParsimPacking(buffer, (int)m.size());
    for (typename std::map<K,V,Tr,A>::const_iterator it = m.begin(); it != m.end(); ++it) {
        doParsimPacking(buffer, it->first);
        doParsimPacking(buffer, it->second);
    }
}

template<typename K, typename V, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::map<K,V,Tr,A>& m)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        K k; V v;
        doParsimUnpacking(buffer, k);
        doParsimUnpacking(buffer, v);
        m[k] = v;
    }
}

// Default pack/unpack function for arrays
template<typename T>
void doParsimArrayPacking(omnetpp::cCommBuffer *b, const T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimPacking(b, t[i]);
}

template<typename T>
void doParsimArrayUnpacking(omnetpp::cCommBuffer *b, T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimUnpacking(b, t[i]);
}

// Default rule to prevent compiler from choosing base class' doParsimPacking() function
template<typename T>
void doParsimPacking(omnetpp::cCommBuffer *, const T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimPacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

template<typename T>
void doParsimUnpacking(omnetpp::cCommBuffer *, T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimUnpacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

}  // namespace omnetpp


// forward
template<typename T, typename A>
std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec);

// Template rule which fires if a struct or class doesn't have operator<<
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const T&) {return out;}

// operator<< for std::vector<T>
template<typename T, typename A>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec)
{
    out.put('{');
    for(typename std::vector<T,A>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        if (it != vec.begin()) {
            out.put(','); out.put(' ');
        }
        out << *it;
    }
    out.put('}');
    
    char buf[32];
    sprintf(buf, " (size=%u)", (unsigned int)vec.size());
    out.write(buf, strlen(buf));
    return out;
}

Register_Class(CtrltoRoutingTableMod)

CtrltoRoutingTableMod::CtrltoRoutingTableMod(const char *name, short kind) : ::omnetpp::cPacket(name,kind)
{
    for (unsigned int i=0; i<100; i++)
        this->ocs0mapleft[i] = 0;
    for (unsigned int i=0; i<100; i++)
        this->ocs0mapright[i] = 0;
    for (unsigned int i=0; i<100; i++)
        this->ocs1mapleft[i] = 0;
    for (unsigned int i=0; i<100; i++)
        this->ocs1mapright[i] = 0;
    this->ocs0mapsize = 0;
    this->ocs1mapsize = 0;
}

CtrltoRoutingTableMod::CtrltoRoutingTableMod(const CtrltoRoutingTableMod& other) : ::omnetpp::cPacket(other)
{
    copy(other);
}

CtrltoRoutingTableMod::~CtrltoRoutingTableMod()
{
}

CtrltoRoutingTableMod& CtrltoRoutingTableMod::operator=(const CtrltoRoutingTableMod& other)
{
    if (this==&other) return *this;
    ::omnetpp::cPacket::operator=(other);
    copy(other);
    return *this;
}

void CtrltoRoutingTableMod::copy(const CtrltoRoutingTableMod& other)
{
    for (unsigned int i=0; i<100; i++)
        this->ocs0mapleft[i] = other.ocs0mapleft[i];
    for (unsigned int i=0; i<100; i++)
        this->ocs0mapright[i] = other.ocs0mapright[i];
    for (unsigned int i=0; i<100; i++)
        this->ocs1mapleft[i] = other.ocs1mapleft[i];
    for (unsigned int i=0; i<100; i++)
        this->ocs1mapright[i] = other.ocs1mapright[i];
    this->ocs0mapsize = other.ocs0mapsize;
    this->ocs1mapsize = other.ocs1mapsize;
}

void CtrltoRoutingTableMod::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::omnetpp::cPacket::parsimPack(b);
    doParsimArrayPacking(b,this->ocs0mapleft,100);
    doParsimArrayPacking(b,this->ocs0mapright,100);
    doParsimArrayPacking(b,this->ocs1mapleft,100);
    doParsimArrayPacking(b,this->ocs1mapright,100);
    doParsimPacking(b,this->ocs0mapsize);
    doParsimPacking(b,this->ocs1mapsize);
}

void CtrltoRoutingTableMod::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::omnetpp::cPacket::parsimUnpack(b);
    doParsimArrayUnpacking(b,this->ocs0mapleft,100);
    doParsimArrayUnpacking(b,this->ocs0mapright,100);
    doParsimArrayUnpacking(b,this->ocs1mapleft,100);
    doParsimArrayUnpacking(b,this->ocs1mapright,100);
    doParsimUnpacking(b,this->ocs0mapsize);
    doParsimUnpacking(b,this->ocs1mapsize);
}

unsigned int CtrltoRoutingTableMod::getOcs0mapleftArraySize() const
{
    return 100;
}

int CtrltoRoutingTableMod::getOcs0mapleft(unsigned int k) const
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    return this->ocs0mapleft[k];
}

void CtrltoRoutingTableMod::setOcs0mapleft(unsigned int k, int ocs0mapleft)
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    this->ocs0mapleft[k] = ocs0mapleft;
}

unsigned int CtrltoRoutingTableMod::getOcs0maprightArraySize() const
{
    return 100;
}

int CtrltoRoutingTableMod::getOcs0mapright(unsigned int k) const
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    return this->ocs0mapright[k];
}

void CtrltoRoutingTableMod::setOcs0mapright(unsigned int k, int ocs0mapright)
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    this->ocs0mapright[k] = ocs0mapright;
}

unsigned int CtrltoRoutingTableMod::getOcs1mapleftArraySize() const
{
    return 100;
}

int CtrltoRoutingTableMod::getOcs1mapleft(unsigned int k) const
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    return this->ocs1mapleft[k];
}

void CtrltoRoutingTableMod::setOcs1mapleft(unsigned int k, int ocs1mapleft)
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    this->ocs1mapleft[k] = ocs1mapleft;
}

unsigned int CtrltoRoutingTableMod::getOcs1maprightArraySize() const
{
    return 100;
}

int CtrltoRoutingTableMod::getOcs1mapright(unsigned int k) const
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    return this->ocs1mapright[k];
}

void CtrltoRoutingTableMod::setOcs1mapright(unsigned int k, int ocs1mapright)
{
    if (k>=100) throw omnetpp::cRuntimeError("Array of size 100 indexed by %lu", (unsigned long)k);
    this->ocs1mapright[k] = ocs1mapright;
}

int CtrltoRoutingTableMod::getOcs0mapsize() const
{
    return this->ocs0mapsize;
}

void CtrltoRoutingTableMod::setOcs0mapsize(int ocs0mapsize)
{
    this->ocs0mapsize = ocs0mapsize;
}

int CtrltoRoutingTableMod::getOcs1mapsize() const
{
    return this->ocs1mapsize;
}

void CtrltoRoutingTableMod::setOcs1mapsize(int ocs1mapsize)
{
    this->ocs1mapsize = ocs1mapsize;
}

class CtrltoRoutingTableModDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertynames;
  public:
    CtrltoRoutingTableModDescriptor();
    virtual ~CtrltoRoutingTableModDescriptor();

    virtual bool doesSupport(omnetpp::cObject *obj) const override;
    virtual const char **getPropertyNames() const override;
    virtual const char *getProperty(const char *propertyname) const override;
    virtual int getFieldCount() const override;
    virtual const char *getFieldName(int field) const override;
    virtual int findField(const char *fieldName) const override;
    virtual unsigned int getFieldTypeFlags(int field) const override;
    virtual const char *getFieldTypeString(int field) const override;
    virtual const char **getFieldPropertyNames(int field) const override;
    virtual const char *getFieldProperty(int field, const char *propertyname) const override;
    virtual int getFieldArraySize(void *object, int field) const override;

    virtual const char *getFieldDynamicTypeString(void *object, int field, int i) const override;
    virtual std::string getFieldValueAsString(void *object, int field, int i) const override;
    virtual bool setFieldValueAsString(void *object, int field, int i, const char *value) const override;

    virtual const char *getFieldStructName(int field) const override;
    virtual void *getFieldStructValuePointer(void *object, int field, int i) const override;
};

Register_ClassDescriptor(CtrltoRoutingTableModDescriptor)

CtrltoRoutingTableModDescriptor::CtrltoRoutingTableModDescriptor() : omnetpp::cClassDescriptor("CtrltoRoutingTableMod", "omnetpp::cPacket")
{
    propertynames = nullptr;
}

CtrltoRoutingTableModDescriptor::~CtrltoRoutingTableModDescriptor()
{
    delete[] propertynames;
}

bool CtrltoRoutingTableModDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<CtrltoRoutingTableMod *>(obj)!=nullptr;
}

const char **CtrltoRoutingTableModDescriptor::getPropertyNames() const
{
    if (!propertynames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
        const char **basenames = basedesc ? basedesc->getPropertyNames() : nullptr;
        propertynames = mergeLists(basenames, names);
    }
    return propertynames;
}

const char *CtrltoRoutingTableModDescriptor::getProperty(const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : nullptr;
}

int CtrltoRoutingTableModDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 6+basedesc->getFieldCount() : 6;
}

unsigned int CtrltoRoutingTableModDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeFlags(field);
        field -= basedesc->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISARRAY | FD_ISEDITABLE,
        FD_ISARRAY | FD_ISEDITABLE,
        FD_ISARRAY | FD_ISEDITABLE,
        FD_ISARRAY | FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
    };
    return (field>=0 && field<6) ? fieldTypeFlags[field] : 0;
}

const char *CtrltoRoutingTableModDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldName(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldNames[] = {
        "ocs0mapleft",
        "ocs0mapright",
        "ocs1mapleft",
        "ocs1mapright",
        "ocs0mapsize",
        "ocs1mapsize",
    };
    return (field>=0 && field<6) ? fieldNames[field] : nullptr;
}

int CtrltoRoutingTableModDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount() : 0;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs0mapleft")==0) return base+0;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs0mapright")==0) return base+1;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs1mapleft")==0) return base+2;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs1mapright")==0) return base+3;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs0mapsize")==0) return base+4;
    if (fieldName[0]=='o' && strcmp(fieldName, "ocs1mapsize")==0) return base+5;
    return basedesc ? basedesc->findField(fieldName) : -1;
}

const char *CtrltoRoutingTableModDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeString(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
    };
    return (field>=0 && field<6) ? fieldTypeStrings[field] : nullptr;
}

const char **CtrltoRoutingTableModDescriptor::getFieldPropertyNames(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldPropertyNames(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

const char *CtrltoRoutingTableModDescriptor::getFieldProperty(int field, const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldProperty(field, propertyname);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

int CtrltoRoutingTableModDescriptor::getFieldArraySize(void *object, int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldArraySize(object, field);
        field -= basedesc->getFieldCount();
    }
    CtrltoRoutingTableMod *pp = (CtrltoRoutingTableMod *)object; (void)pp;
    switch (field) {
        case 0: return 100;
        case 1: return 100;
        case 2: return 100;
        case 3: return 100;
        default: return 0;
    }
}

const char *CtrltoRoutingTableModDescriptor::getFieldDynamicTypeString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldDynamicTypeString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    CtrltoRoutingTableMod *pp = (CtrltoRoutingTableMod *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string CtrltoRoutingTableModDescriptor::getFieldValueAsString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldValueAsString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    CtrltoRoutingTableMod *pp = (CtrltoRoutingTableMod *)object; (void)pp;
    switch (field) {
        case 0: return long2string(pp->getOcs0mapleft(i));
        case 1: return long2string(pp->getOcs0mapright(i));
        case 2: return long2string(pp->getOcs1mapleft(i));
        case 3: return long2string(pp->getOcs1mapright(i));
        case 4: return long2string(pp->getOcs0mapsize());
        case 5: return long2string(pp->getOcs1mapsize());
        default: return "";
    }
}

bool CtrltoRoutingTableModDescriptor::setFieldValueAsString(void *object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->setFieldValueAsString(object,field,i,value);
        field -= basedesc->getFieldCount();
    }
    CtrltoRoutingTableMod *pp = (CtrltoRoutingTableMod *)object; (void)pp;
    switch (field) {
        case 0: pp->setOcs0mapleft(i,string2long(value)); return true;
        case 1: pp->setOcs0mapright(i,string2long(value)); return true;
        case 2: pp->setOcs1mapleft(i,string2long(value)); return true;
        case 3: pp->setOcs1mapright(i,string2long(value)); return true;
        case 4: pp->setOcs0mapsize(string2long(value)); return true;
        case 5: pp->setOcs1mapsize(string2long(value)); return true;
        default: return false;
    }
}

const char *CtrltoRoutingTableModDescriptor::getFieldStructName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructName(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    };
}

void *CtrltoRoutingTableModDescriptor::getFieldStructValuePointer(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructValuePointer(object, field, i);
        field -= basedesc->getFieldCount();
    }
    CtrltoRoutingTableMod *pp = (CtrltoRoutingTableMod *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}


