import Hero from "./Hero"
import Footer from "./HomeComps/Footer"
export const LandingPage = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-grow">
        <Hero />
      </main>
      <Footer />
    </div>
  );
}
