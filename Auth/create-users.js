const axios = require('axios');

const AUTH_URL = 'http://localhost:4000/auth';

async function createUsers() {
  try {
    // User 1: u1 (student)
    console.log('Creating user u1...');
    const user1Response = await axios.post(`${AUTH_URL}/register`, {
      regNo: 'u1',
      email: 'vasudev82090@gmail.com',
      password: '123',
      role: 'user'
    });
    console.log('✅ User u1 created:', user1Response.data);

    // User 2: a1 (admin)
    console.log('Creating admin a1...');
    const user2Response = await axios.post(`${AUTH_URL}/register`, {
      regNo: 'a1',
      email: 'tocrackjee2023@gmail.com',
      password: '123',
      role: 'admin'
    });
    console.log('✅ Admin a1 created:', user2Response.data);

    console.log('\n🎉 Both users created successfully!');
  } catch (error) {
    console.error('❌ Error creating users:', error.response?.data || error.message);
  }
}

createUsers();