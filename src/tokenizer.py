
class CharTokenizer:
      def __init__(self,stories):
          self.char_to_id = {}
          self.id_to_char = {}
          self.vocab_size = 0
          self.unique_characters(stories)

      def character_to_id(self,character_set):
          for index, character in enumerate(character_set, 1):
              self.char_to_id[character] = index
          return self.char_to_id

      def id_to_character(self,result):
          for key, value in result.items():
              self.id_to_char[value] = key
          return self.id_to_char

      def encode(self,text):
          id_list = []
          unknown_character_id = 0
          for char in text:
             id_list.append(self.char_to_id[char] if char in self.char_to_id.keys() else unknown_character_id)
          return id_list

      def decode(self,result):
         decoded_string = ''
         for num in result:
             decoded_string += (self.id_to_char[num] if num != 0 else "?")
         return decoded_string


      def unique_characters(self,stories):
         character_set = set()
         for story in stories:
             for character in story:
                character_set.add(character)
         character_set = sorted(character_set)
         self.vocab_size = len(character_set) + 1
         result = self.character_to_id(character_set)
         self.id_to_character(result)
       

stories = ds['text']
tok = CharTokenizer(stories)
tok.decode(tok.encode(stories[0]))
