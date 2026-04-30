/*aash template*/
#include<bits/stdc++.h>
#define all(v) v.begin(), v.end()
using  ll= long long;
#define vll vector<ll>
#define vi vector<int>
#define ii  pair<ll,ll>
#define vii  vector<ii>
#define fi first
#define sc second
#define sz size()
using namespace std;

const ll INF = 1e9 + 7;
#define FIO ios_base::sync_with_stdio(0);cin.tie(NULL);
template <typename T> 
void cout_v(vector<T> &v){
    for(auto c: v) cout << c << " ";
    cout << endl;
}
template <typename T> 
void fillv(vector<T> &v){
    for(auto& c: v) cin >> c;
}

template <typename T> 
T minv(vector<T> v){
    T mn = (T)INF;
    for(auto c:v) mn = min(mn,c);
    return mn;
}
template <typename T> 
T maxv(vector<T> v){
    T mn = (T)-INF;
    for(auto c:v) mn = max(mn,c);
    return mn;
}
template <typename T> 
T sumv(vector<T> v){
    T sum = (T)(0);
    for(auto c:v) sum+=c;
    return sum;
}

inline vll prefix(vll& v){
    int n = v.sz;
    vll ans(n,0);
    ans[0] = v[0];
    for(int i =1;i< n;i++) ans[i] = ans[i-1]+ v[i];
    return ans;
} 
const ll MOD = 1e9 + 7;
#define nv() int n;cin >> n;vll v(n);fillv(v);

void solve() {
    nv()
}

int main() {
    FIO
    int t=1;
    //cin >> t;
    while (t--) {
        solve();
    }
}