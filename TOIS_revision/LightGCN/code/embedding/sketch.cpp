#include <bits/stdc++.h>

using namespace std;


extern "C" {
    const static int m1 = 4, m2 = 4;
    const double V = 10000;
    double alpha = 1.000001;
    int ins[16384], que[16384];
    double decay_importance = 1;
    int batch_num = 0;
    bool global_flip_bit = 0;
    class SS {
    private:
        int adjust_thres;
        int lim, num, real_n, t = 0;
        int n1, n2;
        float Threshold;
        double tot;
        double p;
        void* addr;
        struct Bucket{
            int val[m1];
            float cnt[m1];
            int dic[m1];
            bool flip_bit;
        }*b;
        struct Bucket2{
            int val[m2];
            float cnt[m2];
            int dic[m2];
            int t[m2];
            bool flip_bit;
        }*b2;
    public:
        queue<uint32_t> hot_id;
        int Hash1(uint32_t x) {
            return x * 1000000007ll % n1;
        }
        int Hash2(uint32_t x) {
            return x * 1000000007ll % n2;
        }
        SS(float Threshold = 200, int lim = 130670, int adjust_thres=1): 
            Threshold(Threshold), lim(lim), adjust_thres(adjust_thres){
            tot = 0;
            num = 0;
            real_n = 0;
            printf("size: %d\n", lim);
            if (addr == NULL) return;
            n1 = lim * 0.9;
            n2 = lim * 0.1;
            b = new Bucket[n1];
            b2 = new Bucket2[n2];
            for (int i = 1; i < lim; ++i)
                hot_id.push(i);
            for (int i = 0; i < n1; ++i) {
                memset(b[i].cnt, 0, sizeof(b[i].cnt));
                memset(b[i].dic, 0, sizeof(b[i].dic));
                memset(b[i].val, 0, sizeof(b[i].val));
                b[i].flip_bit = 0;
            }
            for (int i = 0; i < n2; ++i) {
                memset(b2[i].cnt, 0, sizeof(b2[i].cnt));
                memset(b2[i].dic, 0, sizeof(b2[i].dic));
                memset(b2[i].val, 0, sizeof(b2[i].val));
                memset(b2[i].t, 0, sizeof(b2[i].t));
                b2[i].flip_bit = 0;
            }
        }
        ~SS() {
            // for (int i = 0; i < lim; ++i) 
            //     delete bucket[i];
            // delete bucket[];
        }
        void reset() {
            printf("reset: %d\n", num);
            fflush(stdout);
            vector<pair<float, int*> > vec;
            for (int key = 0; key < n1; ++key) {
                if (b[key].flip_bit != global_flip_bit)
                    update_filp_bit1(key);
                for (int i = 0; i < m1; ++i) {
                    if (b[key].cnt[i] >= Threshold || b[key].dic[i])
                        vec.push_back(make_pair(b[key].cnt[i], &b[key].dic[i]));
                }
            }
            for (int key = 0; key < n2; ++key) {
                if (b2[key].flip_bit != global_flip_bit)
                    update_filp_bit2(key);
                for (int i = 0; i < m2; ++i) {
                    if (b2[key].cnt[i] >= Threshold || b2[key].dic[i])
                        vec.push_back(make_pair(b2[key].cnt[i], &b2[key].dic[i]));
                }
            }
            sort(vec.begin(), vec.end());
            int l = vec.size();
            for (int i = 0; i <= l - lim; ++i) {
                if ((*vec[i].second) != 0) {
                    hot_id.push(*vec[i].second);
                    *vec[i].second = 0;
                }
            }
            printf("reset***: %d, %d, %ld\n", l, lim, hot_id.size());
            fflush(stdout);
            for (int i = l - lim + 1; i < l; ++i) {
                if ((*vec[i].second) == 0 && !hot_id.empty()){
                    *vec[i].second = hot_id.front();
                    hot_id.pop();
                }
            }
            Threshold = vec[l - lim].first;
            cout << "threshold: " << Threshold << endl;
            fflush(stdout);
            real_n = lim;
        }

        int query(uint32_t val) {
            int key = Hash1(val);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == val) {
                    if (b[key].dic[i]) return -b[key].dic[i];
                }
            }
            // int v = queryLRU(val);
            // if (v != 0) return -v;
            return val;
        }

        void update_filp_bit1(int key) {
            b[key].flip_bit = global_flip_bit;
            for (int i = 0; i < m1; ++i)
                b[key].cnt[i] /= V;
        }

        void update_filp_bit2(int key) {
            b2[key].flip_bit = global_flip_bit;
            for (int i = 0; i < m2; ++i)
                b2[key].cnt[i] /= V;
        }

        bool Query(uint32_t x) {
            int key = Hash1(x);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == x) {
                    return 1;
                }
            }
            return 0;
        }

        int queryLRU(uint32_t x) {
            int key = Hash2(x);
            for (int i = 0; i < m2; ++i) {
                if (b2[key].val[i] == x) {
                    return b2[key].dic[i];
                }
            }
            return 0;
        }
        bool insertLRU(uint32_t x, float count = 1) {
            int key = Hash2(x), p = -1, t_min = 1e9, id = 0;
            if (b2[key].flip_bit != global_flip_bit) update_filp_bit2(key);
            for (int i = 0; i < m2; ++i) {
                if (b2[key].val[i] == x) {
                    b2[key].t[i] = ++t;
                    b2[key].cnt[i] += count;
                    // cout << "cnt: " << b2[key].cnt[i] << endl;
                    if (b2[key].cnt[i]>= Threshold && b2[key].cnt[i] - count < Threshold) {
                        real_n ++;
                    }
                    if (b2[key].cnt[i] >= Threshold && !hot_id.empty() && !b2[key].dic[i]) {
                        // cout << key << " " << i << " " << hot_id.front() << " " << b2[key].dic[i] << " " << b2[key].cnt[i] << endl;
                        b2[key].dic[i] = hot_id.front();
                        id = 1;
                        ++num;
                        hot_id.pop();
                    }
                    if (b2[key].cnt[i] >= Threshold) {
                        if (Insert(b2[key].val[i], b2[key].cnt[i], b2[key].dic[i]) == 0) {
                            id = 0;
                        }
                        b2[key].val[i] = 0;
                        b2[key].t[i] = 0;
                        b2[key].cnt[i] = 0;
                        b2[key].dic[i] = 0;
                    }
                    return id;
                }
                if (t_min > b2[key].t[i]) 
                    t_min = b2[key].t[i], p = i;
            }
            if (b2[key].dic[p] || b2[key].cnt[p] >= 5) {
                Insert(b2[key].val[p], b2[key].cnt[p], b2[key].dic[p]);
            }
            b2[key].val[p] = x;
            b2[key].t[p] = ++t;
            b2[key].cnt[p] = 1;
            b2[key].dic[p] = 0;
            if (b2[key].cnt[p]>= Threshold && b2[key].cnt[p] - count < Threshold) {
                real_n ++;
            }
            if (b2[key].cnt[p] >= Threshold && !hot_id.empty() && !b2[key].dic[p]) {
                b2[key].dic[p] = hot_id.front();
                id = 1;
                ++num;
                hot_id.pop();
            }
            return id;
        }
        int insert(uint32_t x, float count = 1) {
            int key = Hash1(x), id = 0;
            if (b[key].flip_bit != global_flip_bit) update_filp_bit1(key);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == x) {
                    b[key].cnt[i] += count;
                    float cnt = b[key].cnt[i];
                    if (cnt >= Threshold && cnt - count < Threshold) {
                        real_n ++;
                    }
                    if (cnt >= Threshold && !b[key].dic[i] && !hot_id.empty()) {
                        b[key].dic[i] = hot_id.front();
                        id = 1;
                        ++num;
                        hot_id.pop();
                    }
                    return id;
                }
            }
            return insertLRU(x, count);
        }
        int Insert(uint32_t x, float count = 1, int Dic = 0) {
            int key = Hash1(x);
            int min_index = -1;
            float min_cnt = 1e9;
            if (b[key].flip_bit != global_flip_bit) update_filp_bit1(key);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == 0) {
                    b[key].val[i] = x;
                    b[key].cnt[i] = count;
                    b[key].dic[i] = Dic;
                    return 1;
                }
                if (b[key].cnt[i] < min_cnt){
                    min_cnt = b[key].cnt[i];
                    min_index = i;
                }
            }
            if (!b[key].dic[min_index]){
                b[key].cnt[min_index] += count;
                b[key].val[min_index] = x;
                b[key].dic[min_index] = Dic;
                return 1;
            } else {
                if (Dic != 0)
                    hot_id.push(Dic);
                return 0;
            }
        }
        int* batch_query(uint32_t *data, int len) {
            for (int i = 0; i < len; ++i) {
                que[i] = query(data[i]);
            }
            return que;
        }
        int* batch_insert(uint32_t *data, int len) {
            ++batch_num;
            decay_importance *= alpha;
            if (decay_importance > V) {
                decay_importance /= V;
                Threshold /= V;
                global_flip_bit ^= 1;
            }
            // cout << "real: " << real_n << " lim: " << lim << " threshold: " << Threshold << endl;
            if (real_n > lim * 1.2 && adjust_thres) reset();
            for (int i = 0; i < len; ++i) {
                ins[i] = insert(data[i], 1);
            }
            return ins;
        }
        int* batch_insert_val(uint32_t *data, float *v, int len) {
            ++batch_num;
            // cout << "real: " << real_n << " " << lim << " " << decay_importance << " " << alpha << endl;
            decay_importance *= alpha;
            if (decay_importance > V) {
                decay_importance /= V;
                Threshold /= V;
                global_flip_bit ^= 1;
            }
            // cout << "real: " << real_n << " lim: " << lim << endl;
            if (real_n > lim * 1.2 && adjust_thres) reset();
            for (int i = 0; i < len; ++i) {
                ins[i] = insert(data[i], v[i]);
            }
            return ins;
        }
    }*ss;
    float cntm[16384];
    
    int* batch_query(uint32_t *data, int len) {
        return ss->batch_query(data, len);
    }
    int* batch_insert(uint32_t *data, int len) {
        return ss->batch_insert(data, len);
    }
    int* batch_insert_val(uint32_t *data, float *v, int len) {
        return ss->batch_insert_val(data, v, len);
    }
    void init(int n, int Threshold, int adjust_thres, double alp){
        ss = new SS((float)Threshold, n, adjust_thres);
        cout << "alp: " << alp << endl;
        alpha = alp;
    }
    double analyse(uint32_t* data, int len) {
        int ans = 0;
        for (int i = 0; i < len; ++i) {
            if (ss->query(data[i]) < 0) ++ans;
        }
        return 1.0 * ans / len;
    }
    double analyse1(uint32_t* data, int len) {
        int ans = 0;
        for (int i = 0; i < len; ++i) {
            if (ss->Query(data[i])) ++ans;
        }
        return 1.0 * ans / len;
    }
}


int main() {
    return 0;
}