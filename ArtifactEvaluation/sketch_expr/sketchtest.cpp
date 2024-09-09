#include <bits/stdc++.h>

using namespace std;


extern "C" {
    int ins[16384], que[16384];
    int bn;
    class SS {
    private:
        int n, k, s, lim, num;
        int Threshold;
        int hotn;
        double tot;
        double p, x, y;
        double alpha;
        void* addr;
        struct Bucket{
            vector<uint32_t> val;
            vector<float> cnt;
            vector<int> dic;
        }*bucket;
        uint32_t Hash(uint32_t val) {
            return (val % s + s) % s;
        }
    public:
        queue<uint32_t> hot_id;
        
        SS(int k = 200, int lim = 130670, int hotn = 130670): k(k), lim(lim), hotn(hotn) {
            Threshold = k;
            s = lim * 4 / bn;
            tot = 0;
            num = 0;
            printf("size: %d\n", s);
            bucket = new Bucket [s];
            n = 0;
            x = 0.25;
            y = 0.25;
            for (int i = 1; i < hotn; ++i)
                hot_id.push(i);
            for (int i = 0; i < s; ++i) {
                bucket[i].cnt.resize(bn);
                bucket[i].dic.resize(bn);
                bucket[i].val.resize(bn);
                for (int j = 0; j < bn; ++j) {
                    bucket[i].cnt[j] = 0;
                    bucket[i].dic[j] = 0;
                    bucket[i].val[j] = 0;
                }
            }
        }
        int query(uint32_t val) {
            int key = Hash(val);
            for (int i = 0; i < bn; ++i) {
                if (bucket[key].cnt[i] != 0 && bucket[key].val[i] == val) {
                    if (bucket[key].dic[i]) return -bucket[key].dic[i];
                }
            }
            return val;
        }
        void print() {
            for (int key = 0; key < s; ++key) {
                for (int i = 0; i < bn; ++i) {
                    cout << bucket[key].val[i] << " " << bucket[key].cnt[i] << " " << bucket[key].dic[i] << endl;
                }
            }
        }
        void update() {
            while(!hot_id.empty()) hot_id.pop();
            bool *v = new bool[s];
            memset(v, 0, sizeof(v));
            for (int key = 0; key < s; ++key) {
                for (int i = 0; i < 4; ++i) {
                    if (bucket[key].dic[i])
                        v[bucket[key].dic[i]] = true;
                }
            }
            for (int i = 1; i < lim; ++i) {
                if (!v[i]) hot_id.push(i);
            }
            delete[] v;
        }
        void decay() {
            printf("decay: hot_nums: %d, tot: %lf %lld\n", num, tot, 1ll * s * k);
            for (int key = 0; key < s; ++key) {
                for (int i = 0; i < bn; ++i) {
                    if (bucket[key].dic[i] && bucket[key].cnt[i] * 0.99 < k){
                        hot_id.push(bucket[key].dic[i]);
                        bucket[key].dic[i] = 0;
                    }
                    bucket[key].cnt[i] *= 0.99;
                }
            }
            tot = 0;
        }
        int Insert(uint32_t val, float v) {
            tot += v;
            int key = Hash(val), id = 0;
            for (int i = 0; i < bn; ++i) {
                if (bucket[key].cnt[i] && bucket[key].val[i] == val) {
                    bucket[key].cnt[i] += v;
                    if (bucket[key].cnt[i] >= k && !hot_id.empty() && !bucket[key].dic[i]) {
                        bucket[key].dic[i] = hot_id.front(), id = 1;
                        hot_id.pop();
                        //printf("%d %d %d %ld\n", key, val, bucket[key].dic[i], hot_id.size());
                        ++num;
                        if (num % 10000 == 0) {
                            printf("num: %d\n", num);
                        }
                    }
                    while(i && bucket[key].cnt[i] > bucket[key].cnt[i-1]) {
                        swap(bucket[key].cnt[i], bucket[key].cnt[i-1]);
                        swap(bucket[key].val[i], bucket[key].val[i-1]);
                        swap(bucket[key].dic[i], bucket[key].dic[i-1]);
                        --i;
                    }
                    return id;
                }
            }
            for (int i = 0; i < bn; ++i) {
                if (bucket[key].cnt[i] == 0) {
                    bucket[key].cnt[i] = v;
                    bucket[key].val[i] = val;
                    return 0;
                }
            }
            if (!bucket[key].dic[bn-1]) {
                bucket[key].cnt[bn-1] += v;
                bucket[key].val[bn-1] = val;
            }
            return 0;
        }
        int* batch_query(uint32_t *data, int len) {
            for (int i = 0; i < len; ++i) {
                que[i] = query(data[i]);
            }
            return que;
        }
        int* batch_insert(uint32_t *data, int len) {
            //printf("%d\n", len);
            if (tot > 1ll * s * k * 10) decay();
            for (int i = 0; i < len; ++i) {
                //printf("%d %d\n", i, data[i]);
                ins[i] = Insert(data[i], 1);
            }
            return ins;
        }
        int* batch_insert_val(uint32_t *data, float *v, int len) {
            //printf("%d\n", len);
            if (tot > 1ll * s * k * 10) decay();
            for (int i = 0; i < len; ++i) {
                //printf("%d %d\n", i, data[i]);
                ins[i] = Insert(data[i], v[i]);
            }
            return ins;
        }
    }*ss;
    float cntm[16384];
    class CUsketch{
    public:
        int k, n;
        float** cnt;
        int* Key;
        CUsketch(int k = 200, int n = 97):k(k), n(n){
            cnt = new float*[k];
            Key = new int[k];
            for (int i = 0; i < k; ++i){
                cnt[i] = new float[n];
                for (int j = 0; j < n; ++j) 
                    cnt[i][j] = 0;
            }
        }
        void insert(uint32_t key, float v) {
            uint32_t p = 998244353;
            int id, mn = 1e9;
            for (int i = 0; i < k; ++i) {
                key = (key + 1) * p;
                Key[i] = key % n;
                if (cnt[i][Key[i]] > mn) 
                    mn = cnt[i][Key[i]],
                    id = i;
                p = p * 998244353u;
            }
            cnt[id][Key[id]] += v;
        }
        float query(uint32_t key) {
            uint32_t p = 998244353;
            float mn = 1e9;
            for (int i = 0; i < k; ++i) {
                key = (key + 1) * p;
                Key[i] = key % n;
                if (cnt[i][key % n] > mn) 
                    mn = cnt[i][Key[i]],
                p = p * 998244353u;
            }
            return mn;
        }
        void batch_insert(uint32_t *data, int len) {
            for (int i = 0; i < len; ++i)
                insert(data[i], 1);
        }
        void batch_insert_val(uint32_t *data, float* v, int len) {
            for (int i = 0; i < len; ++i)
                insert(data[i], v[i]);
        }
        float* batch_cnt(uint32_t *data, int len) {
            for (int i = 0; i < len; ++i)
                cntm[i] = query(data[i]);
            return cntm;
        }
    }CU;
    
    int* batch_query(uint32_t *data, int len) {
        return ss->batch_query(data, len);
    }
    float* batch_cnt(uint32_t *data, int len) {
        return CU.batch_cnt(data, len);
    }
    int* batch_insert(uint32_t *data, int len) {
        return ss->batch_insert(data, len);
    }
    int* batch_insert_val(uint32_t *data, float *v, int len) {
        return ss->batch_insert_val(data, v, len);
    }
    void update() {
        ss->update();
        //ss->print();
    }
    void print() {
        ss->print();
    }
    void init(int n, int Threshold, int hotn, int bucket_n){
        bn = bucket_n;
        ss = new SS(Threshold, n, hotn);
    }
    float analyse(uint32_t *data, int len) {
        float hit = 0, tot = len;
        for (int i = 0; i < len; ++i) {
            if (ss->query(data[i]) < 0) ++hit;
        }
        return hit / tot;
    }
}


int main() {
    return 0;
}