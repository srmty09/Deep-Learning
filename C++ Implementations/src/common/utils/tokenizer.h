#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cctype>

class Tokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    int vocab_size;
    bool is_pretrained;

public:
    Tokenizer() : vocab_size(0), is_pretrained(false) {}
    
    void build_vocab_from_text(const std::string& text) {
        std::unordered_map<std::string, int> char_counts;
        for(char c : text) {
            std::string token(1, c);
            char_counts[token]++;
        }
        
        int id = 0;
        for(const auto& pair : char_counts) {
            token_to_id[pair.first] = id;
            id_to_token[id] = pair.first;
            id++;
        }
        vocab_size = id;
        is_pretrained = false;
    }
    
    void load_pretrained_vocab_json(const std::string& vocab_json_path) {
        std::ifstream file(vocab_json_path);
        if(!file.is_open()) {
            std::cerr << "Warning: Could not load vocab.json from " << vocab_json_path << std::endl;
            return;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        size_t pos = 0;
        while(pos < content.length()) {
            while(pos < content.length() && (std::isspace(content[pos]) || content[pos] == '{' || content[pos] == '}')) pos++;
            if(pos >= content.length()) break;
            
            if(content[pos] == '"') {
                pos++;
                std::string token;
                while(pos < content.length() && content[pos] != '"') {
                    if(content[pos] == '\\' && pos + 1 < content.length()) {
                        pos++;
                        if(content[pos] == 'n') token += '\n';
                        else if(content[pos] == 't') token += '\t';
                        else if(content[pos] == '\\') token += '\\';
                        else if(content[pos] == '"') token += '"';
                        else token += content[pos];
                    } else {
                        token += content[pos];
                    }
                    pos++;
                }
                pos++;
                
                while(pos < content.length() && (std::isspace(content[pos]) || content[pos] == ':')) pos++;
                
                std::string id_str;
                while(pos < content.length() && std::isdigit(content[pos])) {
                    id_str += content[pos];
                    pos++;
                }
                
                if(!id_str.empty()) {
                    int id = std::stoi(id_str);
                    token_to_id[token] = id;
                    id_to_token[id] = token;
                    if(id >= vocab_size) vocab_size = id + 1;
                }
                
                while(pos < content.length() && (std::isspace(content[pos]) || content[pos] == ',')) pos++;
            } else {
                pos++;
            }
        }
        
        is_pretrained = true;
    }
    
    void load_pretrained_vocab(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if(!file.is_open()) {
            std::cerr << "Warning: Could not load pretrained vocab from " << vocab_path << std::endl;
            std::cerr << "Falling back to character-level tokenizer" << std::endl;
            return;
        }
        
        std::string line;
        int id = 0;
        while(std::getline(file, line)) {
            if(!line.empty()) {
                token_to_id[line] = id;
                id_to_token[id] = line;
                id++;
            }
        }
        vocab_size = id;
        is_pretrained = true;
        file.close();
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        if(is_pretrained && !token_to_id.empty()) {
            for(char c : text) {
                std::string token(1, c);
                if(token_to_id.find(token) != token_to_id.end()) {
                    tokens.push_back(token_to_id[token]);
                } else {
                    tokens.push_back(0);
                }
            }
        } else {
            for(char c : text) {
                std::string token(1, c);
                if(token_to_id.find(token) != token_to_id.end()) {
                    tokens.push_back(token_to_id[token]);
                } else {
                    tokens.push_back(0);
                }
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for(int token_id : tokens) {
            if(id_to_token.find(token_id) != id_to_token.end()) {
                text += id_to_token[token_id];
            }
        }
        return text;
    }
    
    int get_vocab_size() const { return vocab_size; }
    bool get_is_pretrained() const { return is_pretrained; }
};

#endif // TOKENIZER_H

