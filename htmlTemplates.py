css = '''
 <style>
 
 
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}


.chat-message.user {
    background-color: #8a9abd
}



.chat-message.bot {
    background-color: #1f48a1
}



.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
  <div class="chat-message bot">
    <div class="message">{{MSG}}</div>
 </div>
'''

user_template = '''
  <div class="chat-message user">  
    <div class="message">{{MSG}}</div>
 </div>
'''